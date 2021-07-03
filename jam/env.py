#coding: utf-8
from os import stat
import time
import numpy as onp
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax.experimental.stax import serial, parallel, Dense, Tanh, Conv, Flatten, FanOut, FanInSum, Identity, BatchNorm
from jax.experimental.optimizers import adam, sgd
from PIL import Image, ImageDraw, ImageOps
from pathlib import Path
from tqdm import tqdm
from enum import IntEnum, auto
from collections import namedtuple, deque
import subprocess
import pickle
from agent import PedestrianAgent
from delay import DelayGen

Experience = namedtuple("Experience",
                        [
                            "state",
                            "action",
                            "reward",
                            "next_state",
                            "next_action",
                            "finished"
                        ]
                        )

class Environment:
    def __init__(self, rng, init_weight_path, batch_size, map_h, map_w, pcpt_h, pcpt_w, max_t, dt, half_decay_dt, n_ped_max):
        self.__rng, rng = jrandom.split(rng)
        self.__batch_size = batch_size
        self.__state_shape = (self.__batch_size, pcpt_h, pcpt_w, EnChannel.num)
        lr = 1E-2
        self.__shared_nn = SharedNetwork(rng, init_weight_path, self.__batch_size, pcpt_h, pcpt_w)
        self.__n_ped_max = n_ped_max
        self.__map_h = map_h
        self.__map_w = map_w
        self.__max_t = max_t
        self.__dt = dt
        self.__agents = None
        self.__experiences = None
        self.__gamma = 0.5 ** (1.0 / (half_decay_dt / self.__dt))
        self.__delay_gen = DelayGen(self.__gamma,  0.5)
        self.__approach_gen = DelayGen(self.__gamma, 0.5)

    @property
    def n_ped_max(self):
        return self.__n_ped_max
    @property
    def map_h(self):
        return self.__map_h
    @property
    def map_w(self):
        return self.__map_w
    @property
    def max_t(self):
        return self.__max_t
    @property
    def dt(self):
        return self.__dt
    @property
    def experiences(self):
        return self.__experiences
    @property
    def policy(self):
        return self.__policy
    @property
    def shared_nn(self):
        return self.__shared_nn
    @property
    def gamma(self):
        return self.__gamma
    @property
    def state_shape(self):
        return self.__state_shape
    def get_agents(self):
        return self.__agents

    def __make_new_pedestrian(self, old_pedestrians):
        while 1:
            rng_y, rng_x, rng_theta, self.__rng = jrandom.split(self.__rng, 4)
            tgt_y, y = jrandom.uniform(rng_y, (2,), minval = 0.0, maxval = self.map_h)
            tgt_x, x = jrandom.uniform(rng_x, (2,), minval = 0.0, maxval = self.map_w)
            theta = jrandom.uniform(rng_theta, (1,), minval = 0.0, maxval = 2.0 * jnp.pi)[0]
            new_ped = PedestrianAgent(tgt_y, tgt_x, y, x, theta, self.dt)

            isolated = True
            if  (new_ped.y >        0.0 + new_ped.radius_m) and \
                (new_ped.y < self.map_h - new_ped.radius_m) and \
                (new_ped.x >        0.0 + new_ped.radius_m) and \
                (new_ped.x < self.map_w - new_ped.radius_m) and \
                (new_ped.tgt_y >        0.0 + new_ped.radius_m) and \
                (new_ped.tgt_y < self.map_h - new_ped.radius_m) and \
                (new_ped.tgt_x >        0.0 + new_ped.radius_m) and \
                (new_ped.tgt_x < self.map_w - new_ped.radius_m):
                for old_ped in old_pedestrians:
                    if  ((new_ped.tgt_y - old_ped.tgt_y) ** 2 + (new_ped.tgt_x - old_ped.tgt_x) ** 2 > (new_ped.radius_m + old_ped.radius_m) ** 2) and \
                        ((new_ped.y     - old_ped.y    ) ** 2 + (new_ped.x     - old_ped.x    ) ** 2 > (new_ped.radius_m + old_ped.radius_m) ** 2):
                        pass
                    else:
                        isolated = False
                        break
            else:
                isolated = False

            if isolated:
                break
        return new_ped
    
    def __make_init_state(self):
        _rng, self.__rng = jrandom.split(self.__rng, 2)
        n_ped = jrandom.randint(_rng, (1,), 1, self.n_ped_max + 1)

        agents = []
        for _ in range(int(n_ped)):
            new_ped = self.__make_new_pedestrian(agents)
            agents.append(new_ped)

        return agents

    def reset(self):
        self.__agents = self.__make_init_state()
        self.__experiences = deque(maxlen = int(self.max_t / self.dt))

    def __step_evolve(self, agent, action):
        accel, omega = action
        y_min = agent.radius_m
        y_max = self.map_h - agent.radius_m
        x_min = agent.radius_m
        x_max = self.map_w - agent.radius_m
        agent.step_evolve(accel, omega, y_min, y_max, x_min, x_max, agent.reached_goal())
        return agent
    
    def __calc_reward(self, agent_idx):
        reward = 0.0
        own = self.__agents[agent_idx]
        # delay
        reward -= self.__delay_gen()
        # hit
        other_agents = self.__agents[:agent_idx] + self.__agents[agent_idx + 1:]
        assert(len(other_agents) == len(self.__agents) - 1)
        for other_agent in other_agents:
            break
            if own.hit_with(other_agent):
                reward += (-1.0)
        # approach
        approach_rate = (jnp.sqrt((own.tgt_y - own.init_y) ** 2 + (own.tgt_x - own.init_x) ** 2) - jnp.sqrt((own.y - own.tgt_y) ** 2 + (own.x - own.tgt_x) ** 2)) / jnp.sqrt(self.__map_h ** 2 + self.__map_w ** 2)
        reward += self.__approach_gen() * approach_rate
        # reach
        if own.reached_goal():
            reward += 1.0
        return reward
    
    def evolve(self):
        for _ in range(int(self.max_t / self.dt)):
            rec = []
            for a in range(len(self.__agents)):
                fin = self.__agents[a].reached_goal()
                state = observe(self.__agents, a, self.__map_h, self.__map_w, self.__state_shape[1], self.__state_shape[2])

                action = self.__agents[a].reserved_action
                if action is None:
                    action = self.__shared_nn.decide_action(state)

                if not fin:
                    self.__agents[a] = self.__step_evolve(self.__agents[a], action)
                    reward = self.__calc_reward(a)
                else:
                    self.__agents[a].stop()
                    reward = 0.0
                
                rec.append((state, action, reward, fin))
            for a in range(len(self.__agents)):
                next_state = observe(self.__agents, a, self.__map_h, self.__map_w, self.__state_shape[1], self.__state_shape[2])
                next_action = self.__shared_nn.decide_action(next_state)
                self.__agents[a].reserved_action = next_action
                state, action, reward, fin = rec[a]
                experience = Experience(state, action, reward, next_state, next_action, fin)
                self.__experiences.append(experience)

            fin_all = True
            for a in range(len(self.__agents)):
                fin_all &= self.__agents[a].reached_goal()
            yield fin_all, self.__agents
            
            if fin_all:
                break
            
class EnAction(IntEnum):
    accel = 0
    omega = auto()
    num = auto()
class EnDist(IntEnum):
    mean = 0
    log_sigma = auto()
    num = auto()

class SharedNetwork:
    def __init__(self, rng, init_weight_path, batch_size, pcpt_h, pcpt_w):
        self.__rng, rng1, rng2, rng3, rng4, rng5 = jrandom.split(rng, 6)

        feature_num = 128
        lr = 1E-3
        SharedNetwork.__state_shape = (batch_size, pcpt_h, pcpt_w, EnChannel.num)
        action_shape = (batch_size, EnAction.num)
        feature_shape = (batch_size, feature_num)

        SharedNetwork.__apply_fun = {}
        self.__opt_init = {}
        SharedNetwork.__opt_update = {}
        SharedNetwork.__get_params = {}
        SharedNetwork.__opt_states = {}
        for k, nn, input_shape, output_num, _rng in [
            ("se", SharedNetwork.state_encoder, SharedNetwork.__state_shape, feature_num, rng1),
            ("ae", SharedNetwork.action_encoder, action_shape, feature_num, rng2),
            ("pd", SharedNetwork.policy_decoder, feature_shape, EnAction.num * EnDist.num, rng3),
            ("vd", SharedNetwork.value_decoder, feature_shape, (1,), rng4),
            ]:
            init_fun, SharedNetwork.__apply_fun[k] = nn(output_num)
            self.__opt_init[k], SharedNetwork.__opt_update[k], SharedNetwork.__get_params[k] = sgd(lr)
            _, init_params = init_fun(_rng, input_shape)
            SharedNetwork.__opt_states[k] = self.__opt_init[k](init_params)
        
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
    def __apply_Pi(params, state):
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
        a_m, a_ls, o_m, o_ls = self.__apply_Pi(SharedNetwork.get_params(self.__opt_states), state)
        self.__rng, rng_a, rng_o = jrandom.split(self.__rng, 3)
        accel = a_m + jnp.exp(a_ls) * jrandom.normal(rng_a)
        omega = o_m + jnp.exp(o_ls) * jrandom.normal(rng_o)
        action = (accel, omega)
        return action
    @staticmethod
    def __log_Pi(params, state, action):
        a = action[:, EnAction.accel]
        o = action[:, EnAction.omega]

        a_m, a_lsig, o_m, o_lsig = SharedNetwork.__apply_Pi(params, state)
        a_sig = jnp.exp(a_lsig)
        o_sig = jnp.exp(o_lsig)
        log_pi = - ((a - a_m) ** 2) / (2 * (a_sig ** 2)) - ((o - o_m) ** 2) / (2 * (o_sig ** 2)) - 2.0 * 0.5 * jnp.log(2 * jnp.pi) - a_lsig - o_lsig
        return log_pi
    @staticmethod
    def __apply_Q(params, state, action):
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
    def __J_q(params, s, a, r, n_s, n_a, gamma):
        next_V = SharedNetwork.__apply_Q(params, n_s, n_a) - SharedNetwork.__log_Pi(params, n_s, n_a)
        return 0.5 * (SharedNetwork.__apply_Q(params, s, a) - (r + gamma * next_V)) ** 2
    @staticmethod
    def __J_pi(params, s, a):
        return SharedNetwork.__log_Pi(params, s, a) - SharedNetwork.__apply_Q(params, s, a)
    @staticmethod
    def __loss(param_se, param_ae, param_pd, param_vd, s, a, r, n_s, n_a, gamma):
        params = {  "se" : param_se,
                    "ae" : param_ae,
                    "pd" : param_pd,
                    "vd" : param_vd,
                    }
        return jnp.mean(SharedNetwork.__J_q(params, s, a, r, n_s, n_a, gamma) + SharedNetwork.__J_pi(params, s, a))
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
            if 0:#jnp.isinf(loss_val1).any() or jnp.isnan(loss_val1).any():
                pass
            else:
                opt_states[k] = SharedNetwork.__opt_update[k](_idx, grad_val, opt_states[k])
            loss_val += loss_val1
        return _idx + 1, opt_states, loss_val
    def update(self, gamma, s, a, r, n_s, n_a):
        self.__learn_cnt, self.__opt_states, loss_val = SharedNetwork.__update(self.__learn_cnt, self.__opt_states, s, a, r, n_s, n_a, gamma)
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

WHITE  = (255, 255, 255)
RED    = (255, 40,    0)
YELLOW = (250, 245,   0)
GREEN  = ( 53, 161, 107)
BLUE   = (  0,  65, 255)
SKY    = (102, 204, 255)
PINK   = (255, 153, 160)
ORANGE = (255, 153,   0)
PURPLE = (154,   0, 121)
BROWN  = (102,  51,   0)
def make_state_img(agents, map_h, map_w, pcpt_h, pcpt_w):
    cols = (WHITE, RED, YELLOW, GREEN, BLUE, SKY, PINK, ORANGE, PURPLE, BROWN)
    img = Image.fromarray(onp.array(jnp.zeros((pcpt_h, pcpt_w, 3), dtype = jnp.uint8)))
    dr = ImageDraw.Draw(img)
    for a, agent in enumerate(agents):
        y = agent.y / map_h * pcpt_h
        x = agent.x / map_w * pcpt_w
        ry = agent.radius_m / map_h * pcpt_h 
        rx = agent.radius_m / map_w * pcpt_w
        
        py0 = jnp.clip(int(y + 0.5), 0, pcpt_h - 1)
        px0 = jnp.clip(int(x + 0.5), 0, pcpt_w - 1)
        py1 = jnp.clip(int((y + ry * onp.sin(agent.theta)) + 0.5), 0, pcpt_h - 1)
        px1 = jnp.clip(int((x + rx * onp.cos(agent.theta)) + 0.5), 0, pcpt_w - 1)
        dr.line((px0, py0, px1, py1), fill = cols[a], width = 1)
        
        py0 = jnp.clip(int((y - ry) + 0.5), 0, pcpt_h - 1)
        py1 = jnp.clip(int((y + ry) + 0.5), 0, pcpt_h - 1)
        px0 = jnp.clip(int((x - rx) + 0.5), 0, pcpt_w - 1)
        px1 = jnp.clip(int((x + rx) + 0.5), 0, pcpt_w - 1)
        dr.ellipse((px0, py0, px1, py1), outline = cols[a], width = 1)

        tgt_y = agent.tgt_y / map_h * pcpt_h 
        tgt_x = agent.tgt_x / map_w * pcpt_w
        tgt_py = jnp.clip(int(tgt_y + 0.5), 0, pcpt_h)
        tgt_px = jnp.clip(int(tgt_x + 0.5), 0, pcpt_w)
        lin_siz = 5
        dr.line((tgt_px - lin_siz, tgt_py - lin_siz, tgt_px + lin_siz, tgt_py + lin_siz), width = 1, fill = cols[a])
        dr.line((tgt_px - lin_siz, tgt_py + lin_siz, tgt_px + lin_siz, tgt_py - lin_siz), width = 1, fill = cols[a])
    dr.rectangle((0, 0, pcpt_w - 1, pcpt_h - 1), outline = WHITE)
    return img

def rotate(y, x, theta):
    rot_x = onp.cos(theta) * x - onp.sin(theta) * y
    rot_y = onp.sin(theta) * x + onp.cos(theta) * y
    return rot_y, rot_x
class EnChannel(IntEnum):
    occupy = 0
    vy = auto()
    vx = auto()
    num = auto()
def observe(agents, agent_idx, map_h, map_w, pcpt_h, pcpt_w):
    state = onp.zeros((pcpt_h, pcpt_w, EnChannel.num), dtype = onp.float32)

    own_obs_y = 0.5 * map_h
    own_obs_x = 0.5 * map_w
    
    own = agents[agent_idx]
    for other in (agents[:agent_idx] + agents[agent_idx + 1:]):
        rel_y = other.y - own.y
        rel_x = other.x - own.x
        obs_y, obs_x = rotate(rel_y, rel_x, 0.5 * onp.pi - own.theta)
        obs_py0 = int((obs_y - other.radius_m + own_obs_y) / map_h * pcpt_h + 0.5)
        obs_py1 = int((obs_y + other.radius_m + own_obs_y) / map_h * pcpt_h + 0.5)
        obs_px0 = int((obs_x - other.radius_m + own_obs_x) / map_w * pcpt_w + 0.5)
        obs_px1 = int((obs_x + other.radius_m + own_obs_x) / map_w * pcpt_w + 0.5)
        if  (obs_py0 <= pcpt_h - 1) and (0 <= obs_py1) and \
            (obs_px0 <= pcpt_w - 1) and (0 <= obs_px1):
            state[max(obs_py0, 0) : min(obs_py1 + 1, pcpt_h), max(obs_px0, 0) : min(obs_px1 + 1, pcpt_w), EnChannel.occupy] = 1.0
            state[max(obs_py0, 0) : min(obs_py1 + 1, pcpt_h), max(obs_px0, 0) : min(obs_px1 + 1, pcpt_w), EnChannel.vy    ] = other.v * onp.cos(own.theta - other.theta)
            state[max(obs_py0, 0) : min(obs_py1 + 1, pcpt_h), max(obs_px0, 0) : min(obs_px1 + 1, pcpt_w), EnChannel.vx    ] = other.v * onp.sin(own.theta - other.theta)

    rel_y = own.tgt_y - own.y
    rel_x = own.tgt_x - own.x
    obs_y, obs_x = rotate(rel_y, rel_x, 0.5 * onp.pi - own.theta)
    obs_py0 = int((obs_y - own.radius_m + own_obs_y) / map_h * pcpt_h + 0.5)
    obs_py1 = int((obs_y + own.radius_m + own_obs_y) / map_h * pcpt_h + 0.5)
    obs_px0 = int((obs_x - own.radius_m + own_obs_x) / map_w * pcpt_w + 0.5)
    obs_px1 = int((obs_x + own.radius_m + own_obs_x) / map_w * pcpt_w + 0.5)
    if  (obs_py0 <= pcpt_h - 1) and (0 <= obs_py1) and \
        (obs_px0 <= pcpt_w - 1) and (0 <= obs_px1):
        state[max(obs_py0, 0) : min(obs_py1 + 1, pcpt_h), max(obs_px0, 0) : min(obs_px1 + 1, pcpt_w), EnChannel.occupy] = - 1.0

    obs_y = 0.0
    obs_x = 0.0
    obs_py0 = int((obs_y - own.radius_m + own_obs_y) / map_h * pcpt_h + 0.5)
    obs_py1 = int((obs_y + own.radius_m + own_obs_y) / map_h * pcpt_h + 0.5)
    obs_px0 = int((obs_x - own.radius_m + own_obs_x) / map_w * pcpt_w + 0.5)
    obs_px1 = int((obs_x + own.radius_m + own_obs_x) / map_w * pcpt_w + 0.5)
    if  (obs_py0 <= pcpt_h - 1) and (0 <= obs_py1) and \
        (obs_px0 <= pcpt_w - 1) and (0 <= obs_px1):
        state[max(obs_py0, 0) : min(obs_py1 + 1, pcpt_h), max(obs_px0, 0) : min(obs_px1 + 1, pcpt_w), EnChannel.vy    ] = own.v
    
    return state

class LogWriter:
    def __init__(self, dst_path):
        if not dst_path.parent.exists():
            dst_path.parent.mkdir(parents = True)
        self.__fp = open(dst_path, "w")
        self.__first = True

    def write(self, out_infos):
        if self.__first:
            LogWriter.__write(self.__fp, True, out_infos)
            self.__first = False
        LogWriter.__write(self.__fp, False, out_infos)

    @staticmethod
    def __write(fp, header, out_infos):
        assert(not (fp is None))
        for i, (key, val) in enumerate(out_infos.items()):
            if header:
                out = key
            else:
                out = val
            fp.write("{},".format(out))
        fp.write("\n")

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst
def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

class Trainer:
    def __init__(self, seed):
        rng = jrandom.PRNGKey(seed)
        self.__rng, rng = jrandom.split(rng)
        batch_size = 128
        map_h = 10.0
        map_w = 10.0
        pcpt_h = 32
        pcpt_w = 32
        max_t = 100.0
        dt = 0.5
        n_ped_max = 4
        half_decay_dt = 10.0
        init_weight_path = None

        self.__env = Environment(rng, init_weight_path, batch_size, map_h, map_w, pcpt_h, pcpt_w, max_t, dt, half_decay_dt, n_ped_max)
    def learn_episode(self, verbose = False):
        log_writer = None
        for trial in range(1000):
            self.__env.reset()

            dst_base_dir = Path("tmp")
            dst_dir = dst_base_dir.joinpath("trial{}".format(trial))
            if not dst_dir.exists():
                dst_dir.mkdir(parents = True)
            dst_png_dir = dst_dir.joinpath("png")
            log_path = dst_base_dir.joinpath("log{}.csv".format(trial))
            if log_writer is not None:
                del log_writer
            log_writer = LogWriter(log_path)

            out_cnt = 0
            total_reward_mean = 0.0
            total_reward = [0.0] * len(self.__env.get_agents())
            if verbose:
                loop_fun = tqdm
            else:
                loop_fun = lambda x : x
            for step, (fin_all, agents) in loop_fun(enumerate(self.__env.evolve())):
                out_infos = {}
                out_infos["step"] = step
                out_infos["t"] = step * self.__env.dt
                for agent_idx, agent in enumerate(agents):
                    experience = self.__env.experiences[-len(agents) + agent_idx]
                    out_infos["tgt_y{}".format(agent_idx)] = agent.tgt_y
                    out_infos["tgt_x{}".format(agent_idx)] = agent.tgt_x
                    out_infos["y{}".format(agent_idx)] = agent.y
                    out_infos["x{}".format(agent_idx)] = agent.x
                    out_infos["v{}".format(agent_idx)] = agent.v
                    out_infos["theta{}".format(agent_idx)] = agent.theta
                    out_infos["accel{}".format(agent_idx)] = experience.action[0]
                    out_infos["omega{}".format(agent_idx)] = experience.action[1]
                    out_infos["reward{}".format(agent_idx)] = experience.reward
                    out_infos["finished{}".format(agent_idx)] = experience.finished
                    total_reward[agent_idx] += experience.reward
                log_writer.write(out_infos)
                
                if (step % int(1.0 / self.__env.dt) == 0) or fin_all:
                    dst_path = dst_png_dir.joinpath("{}.png".format(out_cnt))
                    if not dst_path.exists():
                        if 0:
                            pcpt_h = 128
                            pcpt_w = 128
                            img = ImageOps.flip(make_state_img(agents, self.__env.map_h, self.__env.map_w, pcpt_h, pcpt_w))
                            img = get_concat_h(img, Image.fromarray((255 * ((observe(agents, 0, self.__env.map_h, self.__env.map_w, pcpt_h, pcpt_w)[::-1,:,EnChannel.occupy] + 1.0) / 2)).astype(jnp.uint8)))
                            img = get_concat_h(img, Image.fromarray((255 * ((observe(agents, 0, self.__env.map_h, self.__env.map_w, pcpt_h, pcpt_w)[::-1,:,EnChannel.vy    ] + 1.5) / 3)).astype(jnp.uint8)))
                            img = get_concat_h(img, Image.fromarray((255 * ((observe(agents, 0, self.__env.map_h, self.__env.map_w, pcpt_h, pcpt_w)[::-1,:,EnChannel.vx    ] + 1.5) / 3)).astype(jnp.uint8)))

                            if not dst_path.parent.exists():
                                dst_path.parent.mkdir(parents = True)
                            dr = ImageDraw.Draw(img)
                            dr.text((0,0), "{}".format(step * self.__env.dt), fill = WHITE)
                            if not dst_path.parent.exists():
                                dst_path.parent.mkdir(parents = True)
                            img.save(dst_path)
                    out_cnt += 1
            
            # after episode
            total_reward_mean = onp.array(total_reward).mean()
            state_shape = self.__env.state_shape
            s = onp.zeros(state_shape, dtype = onp.float32)
            a = onp.zeros((state_shape[0], EnAction.num), dtype = onp.float32)
            r = onp.zeros((state_shape[0], ), dtype = onp.float32)
            n_s = onp.zeros(state_shape, dtype = onp.float32)
            n_a = onp.zeros((state_shape[0], EnAction.num), dtype = onp.float32)
            gamma = self.__env.gamma
            val = 0
            total_loss = []
            self.__rng, rng = jrandom.split(self.__rng)
            for i in jrandom.choice(rng, jnp.arange(len(self.__env.experiences)), (len(self.__env.experiences) * 10,)):
                e = self.__env.experiences[i]
                if not e.finished:
                    s[val] = e.state
                    a[val] = e.action
                    r[val] = e.reward
                    n_s[val] = e.next_state
                    n_a[val] = e.next_action
                    val += 1
                    if val >= state_shape[0]:
                        learn_cnt, loss_val = self.__env.shared_nn.update(gamma, s, a, r, n_s, n_a)
                        with open(dst_dir.parent.joinpath("loss.csv"), "a") as f:
                            f.write("{},{},{},{}\n".format(trial, learn_cnt, total_reward_mean, loss_val))
                        if verbose:
                            print("{},{},{},{}".format(trial, learn_cnt, total_reward_mean, loss_val))
                        total_loss.append(loss_val)
                        val = 0
            print("episode={},total_reward_mean={:.3f},loss_mean={:.3f}".format(trial, total_reward_mean, onp.array(total_loss).mean()))
            weight_path = dst_dir.joinpath("param.bin")
            self.__env.shared_nn.save(weight_path)
            #movie_cmd = "cd \"{}\" && ffmpeg -r 10 -i %d.png -vcodec libx264 -pix_fmt yuv420p \"{}\"".format(dst_png_dir.resolve(), dst_png_dir.parent.joinpath("out.mp4").resolve())
            #subprocess.run(movie_cmd)

def main():
    seed = 1
    trainer = Trainer(seed)
    trainer.learn_episode()

if __name__ == "__main__":
    main()