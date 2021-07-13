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

    num = auto()
class SharedNetwork:
    def __init__(self, cfg, rng, init_weight_path, batch_size, pcpt_h, pcpt_w):
        self.__rng, rng_se0, rng_se1, rng_ae0, rng_ae1, rng_vd0, rng_vd1, rng_sep, rng_pdp = jrandom.split(rng, 9)

        feature_num = 128
        self.__log_alpha = 0.0
        SharedNetwork.__state_shape = (batch_size, pcpt_h, pcpt_w, EnChannel.num)
        action_shape = (batch_size, EnAction.num)
        feature_shape = (batch_size, feature_num)

        SharedNetwork.__apply_fun  = [None] * EnModel.num
        SharedNetwork.__opt_init   = [None] * EnModel.num
        SharedNetwork.__opt_update = [None] * EnModel.num
        SharedNetwork.__get_params = [None] * EnModel.num
        self.__opt_states          = [None] * EnModel.num
        q_lr = 1E-3
        p_lr = 1E-3
        SharedNetwork.target_clip = 1.0 / jnp.sqrt(10)
        for i, nn, input_shape, output_num, _rng, lr in [
            (EnModel.q_se0,  SharedNetwork.state_encoder, SharedNetwork.__state_shape, feature_num, rng_se0, q_lr),
            (EnModel.q_se1,  SharedNetwork.state_encoder, SharedNetwork.__state_shape, feature_num, rng_se1, q_lr),
            (EnModel.q_se0t, SharedNetwork.state_encoder, SharedNetwork.__state_shape, feature_num, rng_se0, q_lr),
            (EnModel.q_se1t, SharedNetwork.state_encoder, SharedNetwork.__state_shape, feature_num, rng_se1, q_lr),
            
            (EnModel.q_ae0,  SharedNetwork.action_encoder, action_shape, feature_num, rng_ae0, q_lr),
            (EnModel.q_ae1,  SharedNetwork.action_encoder, action_shape, feature_num, rng_ae1, q_lr),
            (EnModel.q_ae0t, SharedNetwork.action_encoder, action_shape, feature_num, rng_ae0, q_lr),
            (EnModel.q_ae1t, SharedNetwork.action_encoder, action_shape, feature_num, rng_ae1, q_lr),
            
            (EnModel.q_vd0,  SharedNetwork.value_decoder, feature_shape, 1, rng_vd0, q_lr),
            (EnModel.q_vd1,  SharedNetwork.value_decoder, feature_shape, 1, rng_vd1, q_lr),
            (EnModel.q_vd0t, SharedNetwork.value_decoder, feature_shape, 1, rng_vd0, q_lr),
            (EnModel.q_vd1t, SharedNetwork.value_decoder, feature_shape, 1, rng_vd1, q_lr),

            (EnModel.p_se, SharedNetwork.state_encoder, SharedNetwork.__state_shape, feature_num, rng_sep, p_lr),
            (EnModel.p_pd, SharedNetwork.policy_decoder, feature_shape, EnAction.num * EnDist.num, rng_pdp, p_lr),
            ]:
            init_fun, SharedNetwork.__apply_fun[i] = nn(output_num)
            SharedNetwork.__opt_init[i], SharedNetwork.__opt_update[i], SharedNetwork.__get_params[i] = sgd(lr)
            _, init_params = init_fun(_rng, input_shape)
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

        a_mean = nn_out[:, EnAction.accel * EnDist.num + EnDist.mean     ]
        a_lsig = nn_out[:, EnAction.accel * EnDist.num + EnDist.log_sigma]
        o_mean = nn_out[:, EnAction.omega * EnDist.num + EnDist.mean     ]
        o_lsig = nn_out[:, EnAction.omega * EnDist.num + EnDist.log_sigma]
        
        return a_mean, a_lsig, o_mean, o_lsig
    def decide_action(self, state):
        params = SharedNetwork.get_params(self.__opt_states)
        self.__rng, rng = jrandom.split(self.__rng)
        action, log_pi, a_mean, a_sig, o_mean, o_sig = SharedNetwork.__action_and_log_Pi(params, state, rng, False)
        return action, a_mean, a_sig, o_mean, o_sig
    @staticmethod
    def __clip_eps(eps):
        return jnp.clip(eps, - SharedNetwork.target_clip, SharedNetwork.target_clip)
    @staticmethod
    def __action_and_log_Pi(params, state, rng, clip):
        # action
        batch_size = state.shape[0]
        a_mean, a_lsig, o_mean, o_lsig = SharedNetwork.apply_Pi(params, state)
        assert(a_mean.shape == (batch_size,))
        assert(a_lsig.shape == (batch_size,))
        assert(o_mean.shape == (batch_size,))
        assert(o_lsig.shape == (batch_size,))
        rng_a, rng_o = jrandom.split(rng)

        eps_a = jrandom.normal(rng_a, shape = (batch_size,))
        eps_a = jax.lax.cond(clip, SharedNetwork.__clip_eps, lambda x:x, eps_a)
        a_sig = jnp.exp(a_lsig)
        accel = a_mean + a_sig * eps_a
        assert(accel.shape == (batch_size,))

        eps_o = jrandom.normal(rng_o, shape = (batch_size,))
        eps_o = jax.lax.cond(clip, SharedNetwork.__clip_eps, lambda x:x, eps_o)
        o_sig = jnp.exp(o_lsig)
        omega = o_mean + o_sig * eps_o
        assert(omega.shape == (batch_size,))

        action = jnp.append(accel.reshape(batch_size, 1), omega.reshape(batch_size, 1), axis = -1)
        assert(action.shape == (batch_size, EnAction.num))

        log_pi = - ((accel - a_mean) ** 2) / (2 * (a_sig ** 2)) - ((omega - o_mean) ** 2) / (2 * (o_sig ** 2)) - 2.0 * 0.5 * jnp.log(2 * jnp.pi) - a_lsig - o_lsig
        log_pi = log_pi.reshape((batch_size, 1))

        return action, log_pi, a_mean, a_sig, o_mean, o_sig
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
        nn_out = SharedNetwork.__apply_fun[vd](vd_params, se_feature + ae_feature)

        assert(nn_out.shape == (state.shape[0], 1))
        return nn_out

    @staticmethod
    def apply_Q_smaller_target(params, state, action):
        q0_t = SharedNetwork.apply_Q(params, state, action, 0 + 2)
        q1_t = SharedNetwork.apply_Q(params, state, action, 1 + 2)
        q_t = jnp.append(q0_t, q1_t, axis = 1).min(axis = 1, keepdims = True)
        assert(q_t.shape == (state.shape[0], 1))
        return q_t

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
    def Jq(params, log_alpha, s, a, r, n_s, gamma, rng, learned_m):
        n_a, log_pi, a_mean, a_sig, o_mean, o_sig = SharedNetwork.__action_and_log_Pi(params, n_s, rng, clip = True)
        q_t = SharedNetwork.apply_Q_smaller_target(params, n_s, n_a)
        alpha = jnp.exp(log_alpha)
        next_V = q_t - alpha * log_pi
        assert(next_V.shape == (n_s.shape[0], 1))
        r = r.reshape((-1 ,1))
        q = SharedNetwork.apply_Q(params, s, a, learned_m)
        td = q - (r + gamma * next_V)
        j_q = 0.5 * (td ** 2)
        assert(j_q.size == s.shape[0])
        return j_q
    @staticmethod
    def J_pi(params, log_alpha, state, rng):
        action, log_pi, a_mean, a_sig, o_mean, o_sig = SharedNetwork.__action_and_log_Pi(params, state, rng, clip = False)
        q_t = SharedNetwork.apply_Q_smaller_target(params, state, action)
        alpha = jnp.exp(log_alpha)
        j_pi = alpha * log_pi - q_t
        return j_pi
    @staticmethod
    def __pi_loss(params, log_alpha, s, rng):
        j_pi = SharedNetwork.J_pi(params, log_alpha, s, rng)
        loss = jnp.mean(j_pi)
        #for param in params:
        #    loss += 1E-5 * net_maker.weight_decay(param)
        return loss
    @staticmethod
    def __q_loss(params, log_alpha, s, a, r, n_s, gamma, rng, learned_m):
        j_q = SharedNetwork.Jq(params, log_alpha, s, a, r, n_s, gamma, rng, learned_m)
        loss = jnp.mean(j_q)
        #for param in params:
        #    loss += 1E-5 * net_maker.weight_decay(param)
        return loss
    @staticmethod
    def J_alpha(log_alpha, params, state, rng):
        min_entropy = 0.0
        action, log_pi, a_mean, a_sig, o_mean, o_sig = SharedNetwork.__action_and_log_Pi(params, state, rng, clip = False)
        alpha = jnp.exp(log_alpha)
        j_alpha = - alpha * (log_pi + min_entropy)
        return j_alpha
    @staticmethod
    def __alpha_loss(log_alpha, params, s, rng):
        j_alpha = SharedNetwork.J_alpha(log_alpha, params, s, rng)
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
    def __update(q_idx, p_idx, _opt_states, log_alpha, loss_balance, s, a, r, n_s, gamma, rng):
        rng_q0, rng_q1, rng_p, rng_a = jrandom.split(rng, 4)
        rng_qs = (rng_q0, rng_q1)
        params = SharedNetwork.get_params(_opt_states)
        
        q_loss_vals = []
        for m in range(2):
            q_loss_val, q_grad_val = jax.value_and_grad(SharedNetwork.__q_loss)(params, log_alpha, s, a, r, n_s, gamma, rng_qs[m], m)
            q_grad_val = net_maker.recursive_scaling(q_grad_val, loss_balance[0])
            for i in [  EnModel.q_se0,
                        EnModel.q_ae0,
                        EnModel.q_vd0
                        ]:
                p_i = i + m
                _opt_states[p_i] = SharedNetwork.__opt_update[p_i](q_idx, q_grad_val[p_i], _opt_states[p_i])
            q_loss_vals.append(q_loss_val)
        
        pi_loss_val, pi_grad_val = jax.value_and_grad(SharedNetwork.__pi_loss)(params, log_alpha, s, rng_p)
        pi_grad_val = net_maker.recursive_scaling(pi_grad_val, loss_balance[1])
        for i in [  EnModel.p_se,
                    EnModel.p_pd
                    ]:
            _opt_states[i] = SharedNetwork.__opt_update[i](p_idx, pi_grad_val[i], _opt_states[i])
        
        a_loss_val, a_grad_val = jax.value_and_grad(SharedNetwork.__alpha_loss)(log_alpha, params, s, rng_a)
        log_alpha = log_alpha - a_grad_val * 1E-3
        
        q_idx = q_idx + 1
        p_idx = p_idx + 1
        _opt_states = jax.lax.cond((q_idx % 1 == 0), SharedNetwork.__update_target, lambda x:x, _opt_states)

        return q_idx, p_idx, _opt_states, log_alpha, jnp.array(q_loss_vals), pi_loss_val
    def update(self, gamma, s, a, r, n_s):
        loss_balance = jax.nn.softmax(SharedNetwork.__loss_balance)

        rng, self.__rng = jrandom.split(self.__rng)
        self.__q_learn_cnt,     self.__p_learn_cnt, self.__opt_states, self.__log_alpha, loss_val_qs, loss_val_pi = SharedNetwork.__update(
            self.__q_learn_cnt, self.__p_learn_cnt, self.__opt_states, self.__log_alpha, loss_balance, s, a, r, n_s, gamma, rng)

        if self.__q_learn_cnt > 1:
            SharedNetwork.__loss_diffs = SharedNetwork.__loss_diffs.at[0].set(loss_val_qs.mean()  - SharedNetwork.__loss_bufs[0])
            SharedNetwork.__loss_diffs = SharedNetwork.__loss_diffs.at[1].set(loss_val_pi - SharedNetwork.__loss_bufs[1])
            #grad_val = jax.jit(jax.grad(SharedNetwork.__balance_loss))(SharedNetwork.__loss_balance, SharedNetwork.__loss_diffs)
            #SharedNetwork.__loss_balance = SharedNetwork.__loss_balance + 0.5 * grad_val
        SharedNetwork.__loss_bufs = SharedNetwork.__loss_bufs.at[0].set(loss_val_qs.mean())
        SharedNetwork.__loss_bufs = SharedNetwork.__loss_bufs.at[1].set(loss_val_pi)
        
        return self.__q_learn_cnt, self.__p_learn_cnt, jnp.exp(self.__log_alpha), loss_val_qs, loss_val_pi, loss_balance

    @staticmethod
    def state_encoder(output_num):
        return serial(  Conv(4, (3, 3), (1, 1), "SAME"), Tanh, BatchNorm(),
                        Conv(4, (3, 3), (1, 1), "SAME"), Tanh, BatchNorm(),
                        Conv(4, (3, 3), (1, 1), "SAME"), Tanh, BatchNorm(),
                        Flatten,
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