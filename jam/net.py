#coding: utf-8
from pathlib import Path
import jax
import jax.numpy as jnp
import jax.random as jrandom
import pickle
import numpy as onp
from enum import IntEnum, auto
from common import EnChannel, EnAction, EnDist
from jax.experimental.stax import serial, parallel, Dense, Tanh, Conv, Flatten, FanOut, FanInSum, Identity, BatchNorm
from jax.experimental.optimizers import adam, sgd
import sys
sys.path.append("/home/isgsktyktt/work/jax")
from model.maker.model_maker import net_maker

class EnModel(IntEnum):
    q_se0 = 0
    q_se1 = auto()
    q_se0t = auto()
    q_se1t = auto()

    q_ae0 = auto()
    q_ae1 = auto()
    q_ae0t = auto()
    q_ae1t = auto()

    q_vd0 = auto()
    q_vd1 = auto()
    q_vd0t = auto()
    q_vd1t = auto()

    p_se = auto()
    p_pd = auto()

    alpha = auto()

    num = auto()
class SharedNetwork:
    def __init__(self, cfg, rng, init_weight_path, batch_size, pcpt_h, pcpt_w):
        self.__rng, rng_param = jrandom.split(rng)
        rngs = jrandom.split(rng_param, EnModel.num)

        feature_num = 128
        SharedNetwork.__cfg = cfg
        SharedNetwork.__state_shape = (batch_size, pcpt_h, pcpt_w, EnChannel.num)
        action_shape = (batch_size, EnAction.num)
        q_feature_shape = (batch_size, feature_num * 2)
        pi_feature_shape = (batch_size, feature_num)

        SharedNetwork.__apply_fun  = [None] * EnModel.num
        SharedNetwork.__opt_init   = [None] * EnModel.num
        SharedNetwork.__opt_update = [None] * EnModel.num
        SharedNetwork.__get_params = [None] * EnModel.num
        self.__opt_states          = [None] * EnModel.num
        q_lr = 1E-3
        p_lr = 1E-3
        a_lr = 1E-5

        SharedNetwork.target_clip = 1.0 / jnp.sqrt(10)
        for i, nn, input_shape, output_num, rng, lr in [
            (EnModel.q_se0,  SharedNetwork.state_encoder, SharedNetwork.__state_shape, feature_num, rngs[EnModel.q_se0], q_lr),
            (EnModel.q_se1,  SharedNetwork.state_encoder, SharedNetwork.__state_shape, feature_num, rngs[EnModel.q_se1], q_lr),
            (EnModel.q_se0t, SharedNetwork.state_encoder, SharedNetwork.__state_shape, feature_num, rngs[EnModel.q_se0], q_lr),
            (EnModel.q_se1t, SharedNetwork.state_encoder, SharedNetwork.__state_shape, feature_num, rngs[EnModel.q_se1], q_lr),
            
            (EnModel.q_ae0,  SharedNetwork.action_encoder, action_shape, feature_num, rngs[EnModel.q_ae0], q_lr),
            (EnModel.q_ae1,  SharedNetwork.action_encoder, action_shape, feature_num, rngs[EnModel.q_ae1], q_lr),
            (EnModel.q_ae0t, SharedNetwork.action_encoder, action_shape, feature_num, rngs[EnModel.q_ae0], q_lr),
            (EnModel.q_ae1t, SharedNetwork.action_encoder, action_shape, feature_num, rngs[EnModel.q_ae1], q_lr),
            
            (EnModel.q_vd0,  SharedNetwork.value_decoder, q_feature_shape, 1, rngs[EnModel.q_vd0], q_lr),
            (EnModel.q_vd1,  SharedNetwork.value_decoder, q_feature_shape, 1, rngs[EnModel.q_vd1], q_lr),
            (EnModel.q_vd0t, SharedNetwork.value_decoder, q_feature_shape, 1, rngs[EnModel.q_vd0], q_lr),
            (EnModel.q_vd1t, SharedNetwork.value_decoder, q_feature_shape, 1, rngs[EnModel.q_vd1], q_lr),

            (EnModel.p_se, SharedNetwork.state_encoder, SharedNetwork.__state_shape, feature_num, rngs[EnModel.p_se], p_lr),
            (EnModel.p_pd, SharedNetwork.policy_decoder, pi_feature_shape, EnAction.num * EnDist.num, rngs[EnModel.p_pd], p_lr),

            (EnModel.alpha, SharedNetwork.alpha_maker, (batch_size,1), (1,), rngs[EnModel.alpha], a_lr),
            ]:
            init_fun, SharedNetwork.__apply_fun[i] = nn(output_num)
            SharedNetwork.__opt_init[i], SharedNetwork.__opt_update[i], SharedNetwork.__get_params[i] = adam(lr)
            _, init_params = init_fun(rng, input_shape)
            self.__opt_states[i] = SharedNetwork.__opt_init[i](init_params)
        
        if init_weight_path is not None:
            assert(Path(init_weight_path).exists())
            self.__load(init_weight_path)
        self.__q_learn_cnt = 0
        self.__p_learn_cnt = 0
    
        SharedNetwork.__loss_num = 2
        SharedNetwork.__loss_balance = jnp.zeros((SharedNetwork.__loss_num,), dtype = jnp.float32)
        SharedNetwork.__loss_diffs = jnp.zeros((SharedNetwork.__loss_num,), dtype = jnp.float32)
        SharedNetwork.__loss_bufs = jnp.zeros((SharedNetwork.__loss_num,), dtype = jnp.float32)
    @staticmethod
    def alpha_maker(output_num):
        def alpha_init_fun(rng, input_shape):
            log_alpha = jnp.ones((1,), dtype = jnp.float32) * jnp.log(SharedNetwork.__cfg.init_alpha)
            batch_size = (input_shape[0], output_num)
            return batch_size, (log_alpha,)
        def alpha_apply_fun(params, **kwargs):
            log_alpha = jnp.array(params)
            return jnp.exp(log_alpha)
        return alpha_init_fun, alpha_apply_fun
    @staticmethod
    def get_params(opt_states):
        params = [None] * EnModel.num
        for i, (get_params, opt_state) in enumerate(zip(SharedNetwork.__get_params, opt_states)):
            params[i] = get_params(opt_state)
        return params
    @staticmethod
    @jax.jit
    def apply_Pi(params, state):
        se = EnModel.p_se
        se_params = params[se]
        feature = SharedNetwork.__apply_fun[se](se_params, state)

        pd = EnModel.p_pd
        pd_params = params[pd]
        nn_out = SharedNetwork.__apply_fun[pd](pd_params, feature)
        assert(nn_out.shape == (state.shape[0], EnAction.num * EnDist.num))

        return nn_out
    def decide_action(self, state, explore):
        params = SharedNetwork.get_params(self.__opt_states)
        self.__rng, rng = jrandom.split(self.__rng)
        action, log_pi, ex_action, ex_log_pi, means, sigs = SharedNetwork.__action_and_log_Pi(params, state, rng, False)
        if explore:
            return action, means, sigs
        else:
            return ex_action, means, sigs
    @staticmethod
    def __clip_eps(eps):
        return jnp.clip(eps, - SharedNetwork.target_clip, SharedNetwork.target_clip)
    @staticmethod
    @jax.jit
    def __action_and_log_Pi(params, state, rng, clip):
        # action
        nn_out = SharedNetwork.apply_Pi(params, state)
        batch_size, out_num = nn_out.shape
        assert(out_num // 2 == EnAction.num)
        means = nn_out[:, :EnAction.num]
        lsigs = nn_out[:, EnAction.num:]
        lsigs = jnp.log(jax.nn.sigmoid(lsigs))
        assert(means.shape == (batch_size, EnAction.num))
        assert(lsigs.shape == (batch_size, EnAction.num))

        epss = jrandom.normal(rng, shape = (batch_size, EnAction.num))
        epss = jax.lax.cond(clip, SharedNetwork.__clip_eps, lambda x:x, epss)
        sigs = jnp.exp(lsigs)
        action = means + sigs * epss
        exploit_action = means
        assert(action.shape == (batch_size, EnAction.num))
        assert(exploit_action.shape == (batch_size, EnAction.num))

        log_pi = - (((action - means) ** 2) / (2 * (sigs ** 2))).sum(axis = -1) - EnAction.num * 0.5 * jnp.log(2 * jnp.pi) - lsigs.sum(axis = -1)
        log_pi = log_pi.reshape((batch_size, 1))
        action = jnp.tanh(action)
        log_pi = log_pi - jnp.log((1.0 - action * action) + 1E-5).sum(axis = -1, keepdims = True)

        exploit_log_pi = - (((exploit_action - means) ** 2) / (2 * (sigs ** 2))).sum(axis = -1) - EnAction.num * 0.5 * jnp.log(2 * jnp.pi) - lsigs.sum(axis = -1)
        exploit_log_pi = exploit_log_pi.reshape((batch_size, 1))
        exploit_action = jnp.tanh(exploit_action)
        exploit_log_pi = exploit_log_pi - jnp.log((1.0 - exploit_action * exploit_action) + 1E-5).sum(axis = -1, keepdims = True)

        return action, log_pi, exploit_action, exploit_log_pi, means, sigs
    @staticmethod
    def apply_Q(params, state, action, m):
        se = EnModel.q_se0 + m
        se_params = params[se]
        se_feature = SharedNetwork.__apply_fun[se](se_params, state)

        ae = EnModel.q_ae0 + m
        ae_params = params[ae]
        ae_feature = SharedNetwork.__apply_fun[ae](ae_params, action)

        vd = EnModel.q_vd0 + m
        vd_params = params[vd]
        feature = jnp.append(se_feature, ae_feature, axis = -1)
        nn_out = SharedNetwork.__apply_fun[vd](vd_params, feature)

        assert(nn_out.shape == (state.shape[0], 1))
        return nn_out

    @staticmethod
    def __apply_Q_smaller(params, state, action, target):
        q0_t = SharedNetwork.apply_Q(params, state, action, 0 + 2 * int(target))
        q1_t = SharedNetwork.apply_Q(params, state, action, 1 + 2 * int(target))
        q_t = jnp.append(q0_t, q1_t, axis = 1).min(axis = 1, keepdims = True)
        assert(q_t.shape == (state.shape[0], 1))
        return q_t

    def apply_Q_smaller(self, state, action):
        params = SharedNetwork.get_params(self.__opt_states)
        return SharedNetwork.__apply_Q_smaller(params, state, action, False)

    def save(self, weight_path):
        params = SharedNetwork.get_params(self.__opt_states)
        with open(weight_path, 'wb') as f:
            pickle.dump(params, f)
    def __load(self, weight_path):
        with open(weight_path, 'rb') as f:
            params = pickle.load(f)
        for i in range(EnModel.num):
            self.__opt_states[i] = SharedNetwork.__opt_init[i](params[i])
    @staticmethod
    def Jq(params, s, a, r, n_s, n_fin, gamma, rng, learned_m):
        n_a, log_pi, ex_a, ex_log_pi, means, sigs = SharedNetwork.__action_and_log_Pi(params, n_s, rng, clip = True)
        q_t = SharedNetwork.__apply_Q_smaller(params, n_s, n_a, True)
        alpha = SharedNetwork.__apply_fun[EnModel.alpha](params[EnModel.alpha])
        next_V = q_t - alpha * log_pi
        assert(next_V.shape == (n_s.shape[0], 1))
        r = r.reshape((-1 ,1))
        q = SharedNetwork.apply_Q(params, s, a, learned_m)
        td = q - (r + (1.0 - n_fin) * gamma * next_V)
        j_q = 0.5 * (td ** 2)
        assert(j_q.size == s.shape[0])
        return j_q
    @staticmethod
    def J_pi(params, state, rng):
        action, log_pi, ex_a, ex_log_pi, means, sigs = SharedNetwork.__action_and_log_Pi(params, state, rng, clip = False)
        q_t = SharedNetwork.__apply_Q_smaller(params, state, action, False)
        alpha = SharedNetwork.__apply_fun[EnModel.alpha](params[EnModel.alpha])
        j_pi = alpha * log_pi - q_t
        return j_pi
    @staticmethod
    def __pi_loss(params, s, rng):
        j_pi = SharedNetwork.J_pi(params, s, rng)
        loss = jnp.mean(j_pi)
        #for param in params:
        #    loss += 1E-5 * net_maker.weight_decay(param)
        return loss
    @staticmethod
    def __q_loss(params, s, a, r, n_s, n_fin, gamma, rng, learned_m):
        j_q = SharedNetwork.Jq(params, s, a, r, n_s, n_fin, gamma, rng, learned_m)
        # importance samplingもどき
        j_q_max = j_q.max()
        j_q_exp = jnp.exp((j_q - j_q_max))
        j_q_w = j_q_exp / j_q_exp.sum()
        loss = jnp.sum(j_q * j_q_w)
        #for param in params:
        #    loss += 1E-5 * net_maker.weight_decay(param)
        return loss
    @staticmethod
    def J_alpha(args):
        return jax.lax.cond(SharedNetwork.__cfg.alpha_adjust, SharedNetwork.J_alpha_Impl, lambda x: 0.0, args)
    @staticmethod
    def J_alpha_Impl(args):
        params, state, rng = args
        action, log_pi, ex_a, ex_log_pi, means, sigs = SharedNetwork.__action_and_log_Pi(params, state, rng, clip = False)
        alpha = SharedNetwork.__apply_fun[EnModel.alpha](params[EnModel.alpha])
        min_H = SharedNetwork.__cfg.min_entropy
        j_alpha = - alpha * (log_pi + min_H)
        return jnp.mean(j_alpha)
    @staticmethod
    def __alpha_loss(params, s, rng):
        j_alpha = SharedNetwork.J_alpha([params, s, rng])
        loss = jnp.mean(j_alpha)
        #for param in params:
        #    loss += 1E-5 * net_maker.weight_decay(param)
        return loss
    @staticmethod
    def __update_target(opt_states):
        tau = 0.005
        params = SharedNetwork.get_params(opt_states)
        for m in range(2):
            for i in [  EnModel.q_se0,
                        EnModel.q_ae0,
                        EnModel.q_vd0
                        ]:
                src_i = i + m
                tgt_i = i + m + 2
                new_param = net_maker.recursive_linear(params[src_i], params[tgt_i], tau, (1.0 - tau))
                opt_states[tgt_i] = SharedNetwork.__opt_init[tgt_i](new_param)
        return opt_states
    @staticmethod
    def __balance_loss(params, loss_diffs):
        pos_loss_diffs = jnp.maximum(loss_diffs, jnp.zeros(loss_diffs.shape, dtype = jnp.float32))
        loss = (jax.nn.softmax(params) * pos_loss_diffs).mean()
        return loss
    @staticmethod
    @jax.jit
    def __update(q_idx, p_idx, _opt_states, loss_balance, s, a, r, n_s, n_fin, gamma, rng):
        rng_q0, rng_q1, rng_p, rng_a = jrandom.split(rng, 4)
        rng_qs = (rng_q0, rng_q1)
        params = SharedNetwork.get_params(_opt_states)
        
        q_loss_vals = []
        for m in range(2):
            q_loss_val, q_grad_val = jax.value_and_grad(SharedNetwork.__q_loss)(params, s, a, r, n_s, n_fin, gamma, rng_qs[m], m)
            q_grad_val = net_maker.recursive_scaling(q_grad_val, loss_balance[0])
            for i in [  EnModel.q_se0,
                        EnModel.q_ae0,
                        EnModel.q_vd0
                        ]:
                p_i = i + m
                _opt_states[p_i] = SharedNetwork.__opt_update[p_i](q_idx, q_grad_val[p_i], _opt_states[p_i])
            q_loss_vals.append(q_loss_val)
        
        pi_loss_val, pi_grad_val = jax.value_and_grad(SharedNetwork.__pi_loss)(params, s, rng_p)
        pi_grad_val = net_maker.recursive_scaling(pi_grad_val, loss_balance[1])
        for i in [  EnModel.p_se,
                    EnModel.p_pd
                    ]:
            _opt_states[i] = SharedNetwork.__opt_update[i](p_idx, pi_grad_val[i], _opt_states[i])
        
        a_loss_val, a_grad_val = jax.value_and_grad(SharedNetwork.__alpha_loss)(params, s, rng_a)
        _opt_states[EnModel.alpha] = SharedNetwork.__opt_update[EnModel.alpha](p_idx, a_grad_val[EnModel.alpha], _opt_states[EnModel.alpha])
        
        q_idx = q_idx + 1
        p_idx = p_idx + 1
        _opt_states = jax.lax.cond((q_idx % 1 == 0), SharedNetwork.__update_target, lambda x:x, _opt_states)

        return q_idx, p_idx, _opt_states, jnp.array(q_loss_vals), pi_loss_val
    def update(self, gamma, s, a, r, n_s, n_fin):
        loss_balance = jax.nn.softmax(SharedNetwork.__loss_balance)

        rng, self.__rng = jrandom.split(self.__rng)
        self.__q_learn_cnt,     self.__p_learn_cnt, self.__opt_states, loss_val_qs, loss_val_pi = SharedNetwork.__update(
            self.__q_learn_cnt, self.__p_learn_cnt, self.__opt_states, loss_balance, s, a, r, n_s, n_fin, gamma, rng)

        if self.__q_learn_cnt > 1:
            SharedNetwork.__loss_diffs = SharedNetwork.__loss_diffs.at[0].set(loss_val_qs.mean()  - SharedNetwork.__loss_bufs[0])
            SharedNetwork.__loss_diffs = SharedNetwork.__loss_diffs.at[1].set(loss_val_pi - SharedNetwork.__loss_bufs[1])
            #grad_val = jax.jit(jax.grad(SharedNetwork.__balance_loss))(SharedNetwork.__loss_balance, SharedNetwork.__loss_diffs)
            #SharedNetwork.__loss_balance = SharedNetwork.__loss_balance + 0.5 * grad_val
        SharedNetwork.__loss_bufs = SharedNetwork.__loss_bufs.at[0].set(loss_val_qs.mean())
        SharedNetwork.__loss_bufs = SharedNetwork.__loss_bufs.at[1].set(loss_val_pi)
        
        params = SharedNetwork.get_params(self.__opt_states)
        alpha = SharedNetwork.__apply_fun[EnModel.alpha](params[EnModel.alpha])
        #self.__opt_states[EnModel.alpha] = SharedNetwork.__opt_init[EnModel.alpha]((jnp.log(max(float(alpha) - 0.000005, 0.001)),))

        return self.__q_learn_cnt, self.__p_learn_cnt, alpha, loss_val_qs, loss_val_pi, loss_balance

    @staticmethod
    def state_encoder(output_num):
        return serial(  Conv(4, (3, 3), (1, 1), "SAME"), Tanh,# BatchNorm(),
                        Conv(4, (3, 3), (1, 1), "SAME"), Tanh,# BatchNorm(),
                        Conv(4, (3, 3), (1, 1), "SAME"), Tanh,# BatchNorm(),
                        Flatten,
                        Dense(128), Tanh,# BatchNormつけるとなぜか出力が固定値になる,
                        Dense(output_num)
        )
    @staticmethod
    def action_encoder(output_num):
        return serial(  Dense(128), Tanh,# BatchNormつけるとなぜか出力が固定値になる,
                        Dense(output_num)
        )
    @staticmethod
    def policy_decoder(output_num):
        return serial(  Dense(128), Tanh,# BatchNormつけるとなぜか出力が固定値になる
                        Dense(output_num)
        )
    @staticmethod
    def value_decoder(output_num):
        return serial(  Dense(128), Tanh,# BatchNormつけるとなぜか出力が固定値になる
                        Dense(output_num),
        )