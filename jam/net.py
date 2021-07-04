#coding: utf-8
import jax
import jax.numpy as jnp
import jax.random as jrandom
import pickle
from common import *
from jax.experimental.stax import serial, parallel, Dense, Tanh, Conv, Flatten, FanOut, FanInSum, Identity, BatchNorm
from jax.experimental.optimizers import adam, sgd

class SharedNetwork:
    def __init__(self, rng, init_weight_path, batch_size, pcpt_h, pcpt_w):
        self.__rng, rng1, rng2, rng3, rng4, rng5 = jrandom.split(rng, 6)

        feature_num = 128
        lr = 1E-4
        SharedNetwork.__state_shape = (batch_size, pcpt_h, pcpt_w, EnChannel.num)
        action_shape = (batch_size, EnAction.num)
        feature_shape = (batch_size, feature_num)

        SharedNetwork.__apply_fun = {}
        self.__opt_init = {}
        SharedNetwork.__opt_update = {}
        SharedNetwork.__get_params = {}
        SharedNetwork.opt_states = {}
        for k, nn, input_shape, output_num, _rng in [
            ("se", SharedNetwork.state_encoder, SharedNetwork.__state_shape, feature_num, rng1),
            ("ae", SharedNetwork.action_encoder, action_shape, feature_num, rng2),
            ("pd", SharedNetwork.policy_decoder, feature_shape, EnAction.num * EnDist.num, rng3),
            ("vd", SharedNetwork.value_decoder, feature_shape, (1,), rng4),
            ]:
            init_fun, SharedNetwork.__apply_fun[k] = nn(output_num)
            self.__opt_init[k], SharedNetwork.__opt_update[k], SharedNetwork.__get_params[k] = sgd(lr)
            _, init_params = init_fun(_rng, input_shape)
            SharedNetwork.opt_states[k] = self.__opt_init[k](init_params)
        
        if init_weight_path is not None:
            if Path(init_weight_path).exists():
                self.__load(init_weight_path)
        self.__learn_cnt = 0
    @staticmethod
    def get_params(opt_states):
        params = {}
        for k, opt_state in opt_states.items():
            params[k] = SharedNetwork.__get_params[k](opt_state)
        return params
    @staticmethod
    def apply_Pi(params, state):
        if state.ndim != len(SharedNetwork.__state_shape):
            state = state.reshape(tuple([1] + list(state.shape)))
        se_params = params["se"]
        feature = SharedNetwork.__apply_fun["se"](se_params, state)
        pd_params = params["pd"]
        nn_out = SharedNetwork.__apply_fun["pd"](pd_params, feature).flatten()
        a_m =  nn_out[EnAction.accel * EnDist.num + EnDist.mean]
        a_ls = nn_out[EnAction.accel * EnDist.num + EnDist.log_sigma]
        o_m =  nn_out[EnAction.omega * EnDist.num + EnDist.mean]
        o_ls = nn_out[EnAction.omega * EnDist.num + EnDist.log_sigma]
        return a_m, a_ls, o_m, o_ls
    def decide_action(self, state):
        a_m, a_ls, o_m, o_ls = SharedNetwork.apply_Pi(SharedNetwork.get_params(SharedNetwork.opt_states), state)
        self.__rng, rng_a, rng_o = jrandom.split(self.__rng, 3)
        accel = a_m + jnp.exp(a_ls) * jrandom.normal(rng_a)
        omega = o_m + jnp.exp(o_ls) * jrandom.normal(rng_o)
        action = (accel, omega)
        return action
    @staticmethod
    def log_Pi(params, state, action):
        a = action[:, EnAction.accel]
        o = action[:, EnAction.omega]

        a_m, a_lsig, o_m, o_lsig = SharedNetwork.apply_Pi(params, state)
        a_sig = jnp.exp(a_lsig)
        o_sig = jnp.exp(o_lsig)
        log_pi = - ((a - a_m) ** 2) / (2 * (a_sig ** 2)) - ((o - o_m) ** 2) / (2 * (o_sig ** 2)) - 2.0 * 0.5 * jnp.log(2 * jnp.pi) - a_lsig - o_lsig
        return log_pi
    @staticmethod
    def apply_Q(params, state, action):
        se_params = params["se"]
        se_feature = SharedNetwork.__apply_fun["se"](se_params, state)
        ae_params = params["ae"]
        ae_feature = SharedNetwork.__apply_fun["ae"](ae_params, action)
        vd_params = params["vd"]
        nn_out = SharedNetwork.__apply_fun["vd"](vd_params, se_feature + ae_feature)
        return nn_out.flatten()
    def save(self, weight_path):
        params = SharedNetwork.get_params(self.__opt_states)
        with open(weight_path, 'wb') as f:
            pickle.dump(params, f)
    def __load(self, weight_path):
        with open(weight_path, 'rb') as f:
            params = pickle.load(f)
        for k in self.__opt_states.keys():
            self.__opt_states[k] = self.__opt_init[k](params[k])
    @staticmethod
    def J_q(params, s, a, r, n_s, n_a, gamma):
        next_V = SharedNetwork.apply_Q(params, n_s, n_a) - SharedNetwork.log_Pi(params, n_s, n_a)
        return 0.5 * (SharedNetwork.apply_Q(params, s, a) - (r + gamma * next_V)) ** 2
    @staticmethod
    def J_pi(params, s, a):
        return SharedNetwork.log_Pi(params, s, a) - SharedNetwork.apply_Q(params, s, a)
    @staticmethod
    def __loss(param_se, param_ae, param_pd, param_vd, s, a, r, n_s, n_a, gamma):
        params = {  "se" : param_se,
                    "ae" : param_ae,
                    "pd" : param_pd,
                    "vd" : param_vd,
                    }
        j_q = SharedNetwork.J_q(params, s, a, r, n_s, n_a, gamma)
        j_pi = SharedNetwork.J_pi(params, s, a)
        return jnp.mean(j_q + j_pi)
    @staticmethod
    @jax.jit
    def __update(_idx, opt_states, s, a, r, n_s, n_a, gamma):
        params = SharedNetwork.get_params(opt_states)
        param_se = params["se"]
        param_ae = params["ae"]
        param_pd = params["pd"]
        param_vd = params["vd"]
        
        loss_val = 0.0
        for i, k in enumerate(SharedNetwork.__opt_update.keys()):
            loss_val1, grad_val = jax.value_and_grad(SharedNetwork.__loss, argnums = i)(param_se, param_ae, param_pd, param_vd, s, a, r, n_s, n_a, gamma)
            opt_states[k] = SharedNetwork.__opt_update[k](_idx, grad_val, opt_states[k])
            loss_val += loss_val1
        return _idx + 1, opt_states, loss_val
    def update(self, gamma, s, a, r, n_s, n_a):
        self.__learn_cnt, self.__opt_states, loss_val = SharedNetwork.__update(self.__learn_cnt, SharedNetwork.opt_states, s, a, r, n_s, n_a, gamma)
        return self.__learn_cnt, loss_val

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