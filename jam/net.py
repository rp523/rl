#coding: utf-8
from pathlib import Path
import jax
import jax.numpy as jnp
import jax.random as jrandom
import pickle
from enum import IntEnum, auto
from common import *
from jax.experimental.stax import serial, parallel, Dense, Tanh, Conv, Flatten, FanOut, FanInSum, Identity, BatchNorm
from jax.experimental.optimizers import adam, sgd
import sys
sys.path.append("/home/isgsktyktt/work/jax")
from model.maker.model_maker import net_maker

class EnModel(IntEnum):
    q_se = 0
    q_ae = auto()
    q_vd = auto()
    p_se = auto()
    p_pd = auto()
    num = auto()
class SharedNetwork:
    def __init__(self, cfg, rng, init_weight_path, batch_size, pcpt_h, pcpt_w):
        self.__rng, rng1, rng2, rng3, rng4, rng5 = jrandom.split(rng, 6)

        feature_num = 128
        SharedNetwork.__temperature = cfg.temperature
        SharedNetwork.__state_shape = (batch_size, pcpt_h, pcpt_w, EnChannel.num)
        action_shape = (batch_size, EnAction.num)
        feature_shape = (batch_size, feature_num)

        SharedNetwork.__apply_fun = [None] * EnModel.num
        self.__opt_init = [None] * EnModel.num
        SharedNetwork.__opt_update = [None] * EnModel.num
        SharedNetwork.__get_params = [None] * EnModel.num
        self.__opt_states = [None] * EnModel.num
        q_lr = 1E-4
        p_lr = 1E-3
        for i, nn, input_shape, output_num, _rng, lr in [
            (EnModel.q_se, SharedNetwork.state_encoder, SharedNetwork.__state_shape, feature_num, rng1, q_lr),
            (EnModel.q_ae, SharedNetwork.action_encoder, action_shape, feature_num, rng2, q_lr),
            (EnModel.q_vd, SharedNetwork.value_decoder, feature_shape, (1,), rng3, q_lr),
            (EnModel.p_se, SharedNetwork.state_encoder, SharedNetwork.__state_shape, feature_num, rng4, p_lr),
            (EnModel.p_pd, SharedNetwork.policy_decoder, feature_shape, EnAction.num * EnDist.num, rng5, p_lr),
            ]:
            init_fun, SharedNetwork.__apply_fun[i] = nn(output_num)
            self.__opt_init[i], SharedNetwork.__opt_update[i], SharedNetwork.__get_params[i] = adam(lr)
            _, init_params = init_fun(_rng, input_shape)
            self.__opt_states[i] = self.__opt_init[i](init_params)
        
        if init_weight_path is not None:
            assert(Path(init_weight_path).exists())
            self.__load(init_weight_path)
        self.__q_learn_cnt = 0
        self.__p_learn_cnt = 0
    
        SharedNetwork.__loss_num = 2
        SharedNetwork.__loss_balance = jnp.zeros((SharedNetwork.__loss_num,), dtype = jnp.float32)
        SharedNetwork.__loss_diffs = jnp.zeros((SharedNetwork.__loss_num,), dtype = jnp.float32)
    @staticmethod
    def get_params(opt_states):
        params = [None] * EnModel.num
        for i, (get_params, opt_state) in enumerate(zip(SharedNetwork.__get_params, opt_states)):
            params[i] = get_params(opt_state)
        return params
    @staticmethod
    @jax.jit
    def apply_Pi(params, state):
        se_params = params[EnModel.p_se]
        feature = SharedNetwork.__apply_fun[EnModel.p_se](se_params, state)
        pd_params = params[EnModel.p_pd]
        nn_out = SharedNetwork.__apply_fun[EnModel.p_pd](pd_params, feature)
        assert(nn_out.shape == (state.shape[0], EnAction.num * EnDist.num))
        a_mean =  nn_out[:,EnAction.accel * EnDist.num + EnDist.mean]
        a_lsig = nn_out[:,EnAction.accel * EnDist.num + EnDist.log_sigma]
        o_mean =  nn_out[:,EnAction.omega * EnDist.num + EnDist.mean]
        o_lsig = nn_out[:,EnAction.omega * EnDist.num + EnDist.log_sigma]
        return a_mean, a_lsig, o_mean, o_lsig
    def decide_action(self, state):
        params = SharedNetwork.get_params(self.__opt_states)
        self.__rng, rng = jrandom.split(self.__rng)
        action, log_pi = SharedNetwork.__action_and_log_Pi(params, state, rng)
        return action
    @staticmethod
    def __action_and_log_Pi(params, state, rng):
        # action
        batch_size = state.shape[0]
        a_mean, a_lsig, o_mean, o_lsig = SharedNetwork.apply_Pi(params, state)
        assert(a_mean.shape == (batch_size,))
        assert(a_lsig.shape == (batch_size,))
        assert(o_mean.shape == (batch_size,))
        assert(o_lsig.shape == (batch_size,))
        rng_a, rng_o = jrandom.split(rng)
        accel = a_mean + jnp.exp(a_lsig) * jrandom.normal(rng_a, shape = (batch_size,))
        omega = o_mean + jnp.exp(o_lsig) * jrandom.normal(rng_o, shape = (batch_size,))
        assert(accel.shape == (batch_size,))
        assert(omega.shape == (batch_size,))
        action = jnp.append(accel.reshape(batch_size, 1), omega.reshape(batch_size, 1), axis = -1)
        assert(action.shape == (batch_size, EnAction.num))

        a_sig = jnp.exp(a_lsig)
        o_sig = jnp.exp(o_lsig)
        log_pi = - ((accel - a_mean) ** 2) / (2 * (a_sig ** 2)) - ((omega - o_mean) ** 2) / (2 * (o_sig ** 2)) - 2.0 * 0.5 * jnp.log(2 * jnp.pi) - a_lsig - o_lsig
        log_pi = (SharedNetwork.__temperature * log_pi).reshape((batch_size, 1))

        return action, log_pi
    @staticmethod
    def apply_Q(params, state, action):
        se_params = params[EnModel.q_se]
        se_feature = SharedNetwork.__apply_fun[EnModel.q_se](se_params, state)
        ae_params = params[EnModel.q_ae]
        ae_feature = SharedNetwork.__apply_fun[EnModel.q_ae](ae_params, action)
        vd_params = params[EnModel.q_vd]
        nn_out = SharedNetwork.__apply_fun[EnModel.q_vd](vd_params, se_feature + ae_feature)
        assert(nn_out.shape == (state.shape[0], 1))
        return nn_out
    def save(self, weight_path):
        params = SharedNetwork.get_params(self.__opt_states)
        with open(weight_path, 'wb') as f:
            pickle.dump(params, f)
    def __load(self, weight_path):
        with open(weight_path, 'rb') as f:
            params = pickle.load(f)
        for k in self.__opt_states.keys():
            SharedNetwork.opt_states[k] = self.__opt_init[k](params[k])
    @staticmethod
    def J_q(params, s, a, r, n_s, gamma, rng):
        n_a, log_pi = SharedNetwork.__action_and_log_Pi(params, n_s, rng)
        next_V = SharedNetwork.apply_Q(params, n_s, n_a) - log_pi
        assert(next_V.size == s.shape[0])
        r = r.reshape((-1 ,1))
        td = SharedNetwork.apply_Q(params, s, a) - (r + gamma * next_V)
        out = 0.5 * td ** 2
        assert(out.size == s.shape[0])
        return out
    @staticmethod
    def J_pi(params, state, rng):
        action, log_pi = SharedNetwork.__action_and_log_Pi(params, state, rng)
        return log_pi - SharedNetwork.apply_Q(params, state, action)
    @staticmethod
    def __pi_loss(params, s, rng):
        j_pi = SharedNetwork.J_pi(params, s, rng)
        loss = jnp.mean(j_pi)
        #for param in params:
        #    loss += 1E-5 * net_maker.weight_decay(param)
        return loss
    @staticmethod
    def __q_loss(params, s, a, r, n_s, gamma, rng):
        j_q = SharedNetwork.J_q(params, s, a, r, n_s, gamma, rng)
        loss = jnp.mean(j_q)
        #for param in params:
        #    loss += 1E-5 * net_maker.weight_decay(param)
        return loss
    @staticmethod
    def __balance_loss(params, loss_diffs):
        pos_loss_diffs = jnp.clip(loss_diffs, 0.0, loss_diffs)
        loss = (jax.nn.softmax(params) * pos_loss_diffs).mean()
        return loss
    @staticmethod
    @jax.jit
    def __update(q_idx, p_idx, _opt_states, loss_balance, s, a, r, n_s, gamma, rng):
        rng_q, rng_p = jrandom.split(rng)
        params = SharedNetwork.get_params(_opt_states)
        
        loss_val_q, grad_val = jax.value_and_grad(SharedNetwork.__q_loss)(params, s, a, r, n_s, gamma, rng_q)
        grad_val = net_maker.recursive_scaling(grad_val, loss_balance[0])
        for i in [EnModel.q_se, EnModel.q_ae, EnModel.q_vd]:
            _opt_states[i] = SharedNetwork.__opt_update[i](q_idx, grad_val[i], _opt_states[i])
        q_idx = q_idx + 1
        
        loss_val_pi, grad_val = jax.value_and_grad(SharedNetwork.__pi_loss)(params, s, rng_p)
        grad_val = net_maker.recursive_scaling(grad_val, loss_balance[1])
        for i in [EnModel.p_se, EnModel.p_pd]:
            _opt_states[i] = SharedNetwork.__opt_update[i](p_idx, grad_val[i], _opt_states[i])
        p_idx = p_idx + 1
        
        return q_idx, p_idx, _opt_states, loss_val_q, loss_val_pi
    def update(self, gamma, s, a, r, n_s):
        #if self.__q_learn_cnt >= 2:
        #    grad_val = jax.jit(jax.grad(SharedNetwork.__balance_loss))(SharedNetwork.__loss_balance, SharedNetwork.__loss_diffs)
        #    SharedNetwork.__loss_balance = SharedNetwork.__loss_balance - 0.1 * grad_val
        loss_balance = jax.nn.softmax(SharedNetwork.__loss_balance)

        rng, self.__rng = jrandom.split(self.__rng)
        self.__q_learn_cnt, self.__p_learn_cnt, self.opt_states, loss_val_q, loss_val_pi = SharedNetwork.__update(self.__q_learn_cnt, self.__p_learn_cnt, self.__opt_states, loss_balance, s, a, r, n_s, gamma, rng)

        if self.__q_learn_cnt == 1:
            SharedNetwork.__loss_diffs = SharedNetwork.__loss_diffs.at[0].set(loss_val_q)
            SharedNetwork.__loss_diffs = SharedNetwork.__loss_diffs.at[1].set(loss_val_pi)
        else:
            SharedNetwork.__loss_diffs = SharedNetwork.__loss_diffs.at[0].set(loss_val_q  - SharedNetwork.__loss_diffs[0])
            SharedNetwork.__loss_diffs = SharedNetwork.__loss_diffs.at[1].set(loss_val_pi - SharedNetwork.__loss_diffs[1])
        
        return self.__q_learn_cnt, self.__p_learn_cnt, loss_val_q, loss_val_pi, loss_balance

    @staticmethod
    def state_encoder(output_num):
        return serial(  Conv( 8, (7, 7), (1, 1), "VALID"), Tanh,
                        Conv(16, (5, 5), (1, 1), "VALID"), Tanh,
                        Conv(16, (3, 3), (1, 1), "VALID"), Tanh,
                        Conv(32, (3, 3), (1, 1), "VALID"), Tanh,
                        Conv(32, (3, 3), (1, 1), "VALID"), Tanh,
                        Flatten,
                        Dense(output_num)
        )
    @staticmethod
    def action_encoder(output_num):
        return serial(  Dense(128), Tanh,
                        Dense(128), Tanh,
                        Dense(128), Tanh,
                        Dense(output_num)
        )
    @staticmethod
    def policy_decoder(output_num):
        return serial(  Dense(128), Tanh,
                        Dense(128), Tanh,
                        Dense(128), Tanh,
                        Dense(output_num)
        )
    def value_decoder(output_num):
        return serial(  Dense(128), Tanh,
                        Dense(128), Tanh,
                        Dense(128), Tanh,
                        Dense(1),
                        Flatten
        )