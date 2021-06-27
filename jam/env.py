#coding: utf-8
import time
import numpy as onp
import jax
import jax.numpy as jnp
import jax.random as jrandom
from PIL import Image, ImageDraw, ImageOps
from pathlib import Path
from tqdm import tqdm
from enum import IntEnum, auto
from collections import namedtuple, deque
import subprocess
from agent import PedestrianAgent
from delay import DelayGen

Experience = namedtuple("Experience",
                        [
                            "state",
                            "action",
                            "reward",
                            "next_state",
                            "finished"
                        ]
                        )

class Environment:
    def __init__(self, rng, map_h, map_w, pcpt_h, pcpt_w, max_t, dt, half_decay_dt, n_ped_max):
        self.__rng, rng = jrandom.split(rng)
        self.__batch_size = 64
        self.__state_shape = (self.__batch_size, pcpt_h, pcpt_w, EnChannel.num)
        lr = 1E-2
        self.__policy = Policy(rng, map_h, map_w, nn_model(EnAction.num), self.__state_shape, lr)
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
            if own.hit_with(other_agent):
                reward += (-1.0)
        # approach
        approach_rate = jnp.sqrt((own.y - own.init_y) ** 2 + (own.x - own.init_x) ** 2) / jnp.sqrt(self.__map_h ** 2 + self.__map_w ** 2)
        reward += self.__approach_gen() * approach_rate
        # reach
        if own.reached_goal():
            reward += 1.0
        return reward
    
    def evolve(self):
        for _ in range(int(self.max_t / self.dt)):
            for a in range(len(self.__agents)):
                fin = self.__agents[a].reached_goal()
                state = observe(self.__agents, a, self.__map_h, self.__map_w, self.__state_shape[1], self.__state_shape[2])
                action = self.__policy(state)
                reward = 0.0
                if not fin:
                    # action
                    # update state
                    self.__agents[a] = self.__step_evolve(self.__agents[a], action)
                    next_state = observe(self.__agents, a, self.__map_h, self.__map_w, self.__state_shape[1], self.__state_shape[2])
                    # reward
                    reward = self.__calc_reward(a)
                else:
                    self.__agents[a].stop()
                    next_state = state.copy()
                experience = Experience(state, action, reward, next_state, fin)
                self.__experiences.append(experience)

            fin_all = True
            for a in range(len(self.__agents)):
                fin_all &= self.__agents[a].reached_goal()
            yield fin_all, self.__agents
            
            if fin_all:
                break
            
class EnAction(IntEnum):
    accel_mean = 0
    accel_log_sigma = auto()
    omega_mean = auto()
    omega_log_sigma = auto()
    num = auto()

def nn_model(output_num):
    return serial(  Conv( 8, (7, 7), (1, 1), "VALID"), Tanh,
                    Conv(16, (5, 5), (1, 1), "VALID"), Tanh,
                    Conv(16, (3, 3), (1, 1), "VALID"), Tanh,
                    Conv(32, (3, 3), (1, 1), "VALID"), Tanh,
                    Conv(32, (3, 3), (1, 1), "VALID"), Tanh,
                    Flatten,
                    Dense(64), Tanh,
                    Dense(output_num)
    )

class Policy:
    def __init__(self, rng, map_h, map_w, nn, state_shape, lr):
        self.__map_h = map_h
        self.__map_w = map_w
        self.__rng, rng_param = jax.random.split(rng)
        init_fun, self.__apply_fun = nn
        opt_init, self.__opt_update, self.__get_params = adam(lr)
        _, init_params = init_fun(rng_param, state_shape)
        self.__opt_state = opt_init(init_params)
        self.__learn_cnt = 0
    
    def __call__(self, obs_state):
        # set action
        self.__rng, rng_a, rng_o = jrandom.split(self.__rng, 3)
        '''
        if 0: #random
            accel = 1.0 * 1.0 * jrandom.normal(rng_a)
            omega = 0.0 + jnp.pi * jrandom.normal(rng_o)
        elif 0: #best
            accel = 1.0
            tgt_theta = jnp.arctan2((agents[agent_idx].tgt_y - agents[agent_idx].y), (agents[agent_idx].tgt_x - agents[agent_idx].x))
            omega = (tgt_theta - agents[agent_idx].theta)
        else:
        '''
        obs_state = obs_state.reshape(tuple([1] + list(obs_state.shape)))
        params = self.__get_params(self.__opt_state)
        nn_out = self.__apply_fun(params, obs_state)[0]

        a_m = nn_out[EnAction.accel_mean]
        a_ls = nn_out[EnAction.accel_log_sigma]
        o_m = nn_out[EnAction.omega_mean]
        o_ls = nn_out[EnAction.omega_log_sigma]
        accel = a_m + jnp.exp(a_ls) * jrandom.normal(rng_a)
        omega = o_m + jnp.exp(o_ls) * jrandom.normal(rng_o)
        return (accel, omega)
    def __loss(self, params, x, y):
        focal_gamma = 2.0
        y_pred = self.__apply_fun(params, x)
        y_pred = jax.nn.softmax(y_pred)
        loss = (- y * ((1.0 - y_pred) ** focal_gamma) * jnp.log(y_pred + 1E-10)).sum(axis = -1).mean()
        return loss
    def learn(self, x, y):
        @jax.jit
        def _update(_idx, _opt_state, _x, _y):
            params = self.__get_params(_opt_state)
            loss_val, grad_val = jax.value_and_grad(self.__loss)(params, _x, _y)
            _opt_state = self.__opt_update(_idx, grad_val, _opt_state)
            return _idx + 1, _opt_state, loss_val
        self.__learn_cnt, self.__opt_state, loss_val = _update(self.__learn_cnt, self.__opt_state, x, y)

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
        _rng, rng = jrandom.split(rng, 2)
        map_h = 10.0
        map_w = 10.0
        pcpt_h = 32
        pcpt_w = 32
        max_t = 1000.0
        dt = 0.5
        n_ped_max = 4
        half_decay_dt = 10.0

        self.__env = Environment(_rng, map_h, map_w, pcpt_h, pcpt_w, max_t, dt, half_decay_dt, n_ped_max)
    def learn_episode(self):
        for trial in range(1):
            self.__env.reset()

            dst_dir = Path("tmp/trial{}".format(trial))
            if not dst_dir.exists():
                dst_dir.mkdir(parents = True)
            dst_png_dir = dst_dir.joinpath("png")
            log_path = dst_dir.joinpath("log.csv")
            log_writer = LogWriter(log_path)

            step = 0
            out_cnt = 0
            for (fin, agents) in tqdm(self.__env.evolve()):
                out_infos = {}
                out_infos["step"] = step
                out_infos["t"] = step * self.__env.dt
                for a, agent in enumerate(agents):
                    experience = self.__env.experiences[-len(agents) + a]
                    out_infos["tgt_y{}".format(a)] = agent.tgt_y
                    out_infos["tgt_x{}".format(a)] = agent.tgt_x
                    out_infos["y{}".format(a)] = agent.y
                    out_infos["x{}".format(a)] = agent.x
                    out_infos["v{}".format(a)] = agent.v
                    out_infos["theta{}".format(a)] = agent.theta
                    out_infos["accel{}".format(a)] = experience.action[0]
                    out_infos["omega{}".format(a)] = experience.action[1]
                    out_infos["reward{}".format(a)] = experience.reward
                    out_infos["finished{}".format(a)] = experience.finished
                log_writer.write(out_infos)
                
                if (step % int(1.0 / self.__env.dt) == 0) or fin:
                    dst_path = dst_png_dir.joinpath("{}.png".format(out_cnt))
                    if not dst_path.exists():
                        if 1:
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

                step += 1
        movie_cmd = "cd \"{}\" && ffmpeg -r 10 -i %d.png -vcodec libx264 -pix_fmt yuv420p \"{}\"".format(dst_png_dir.resolve(), dst_png_dir.parent.joinpath("out.mp4").resolve())
        print(movie_cmd)
        print(subprocess.getoutput(movie_cmd))

def main():
    seed = 1
    trainer = Trainer(seed)
    trainer.learn_episode()

#coding: utf-8
import subprocess
import sys
sys.path.append("jax")
import time
from pathlib import Path
from PIL import Image
from enum import IntEnum, auto, unique
import jax
import jax.numpy as jnp
from jax.experimental.stax import serial, parallel, Dense, Tanh, Conv, Flatten, FanOut, FanInSum, Identity, BatchNorm
from jax.experimental.optimizers import adam
from dataset.fashion_mnist import FashionMnist

def nn(cn):
    return serial(  Conv( 8, (7, 7), (1, 1), "VALID"), Tanh,
                    Conv(16, (5, 5), (1, 1), "VALID"), Tanh,
                    Conv(16, (3, 3), (1, 1), "VALID"), Tanh,
                    Conv(32, (3, 3), (1, 1), "VALID"), Tanh,
                    Conv(32, (3, 3), (1, 1), "VALID"), Tanh,
                    Flatten,
                    Dense(64), Tanh,
                    Dense(2)
    )

class MnistTrainer:
    def __init__(self, rng):
        self.__rng = rng
        self.__rng_train, self.__rng_test, rng_param = jax.random.split(self.__rng, 3)
        self.__batch_size = 128
        class_num = 10
        lr = 1E-4

        init_fun, self.__apply_fun = nn(class_num)
        opt_init, self.__opt_update, self.__get_params = adam(lr)
        input_shape = (self.__batch_size, 28, 28, 1)
        _, init_params = init_fun(rng_param, input_shape)
        self.__opt_state = opt_init(init_params)

    def __loss(self, params, x, y):
        focal_gamma = 2.0
        y_pred = self.__apply_fun(params, x)
        y_pred = jax.nn.softmax(y_pred)
        loss = (- y * ((1.0 - y_pred) ** focal_gamma) * jnp.log(y_pred + 1E-10)).sum(axis = -1).mean()
        return loss
    def learn(self):
        @jax.jit
        def update(_idx, _opt_state, _x, _y):
            params = self.__get_params(_opt_state)
            loss_val, grad_val = jax.value_and_grad(self.__loss)(params, _x, _y)
            _opt_state = self.__opt_update(_idx, grad_val, _opt_state)
            return _idx + 1, _opt_state, loss_val
        tmp_dir_path = "tmp"
        train_data = FashionMnist(
            rng = self.__rng_train,
            batch_size = self.__batch_size,
            data_type = "train",
            one_hot = True,
            dequantize = True,
            flatten = False,
            dir_path = tmp_dir_path,
        )
        idx = 0
        run_loss = 0.0
        run_cnt = 0
        epoch_cnt = 0.0
        reach_epoch = 0
        while True:
            x, y = train_data.sample()

            idx, self.__opt_state, loss_val = update(idx, self.__opt_state, x, y)

            run_loss += loss_val
            run_cnt += 1
            epoch_cnt += (self.__batch_size) / 60000
            
            if epoch_cnt >= reach_epoch:
                reach_epoch += 1

                cmd = "{},{}".format(
                    epoch_cnt,
                    run_loss / run_cnt
                )
                print(cmd)
                run_loss = 0.0
                run_cnt = 0

                if reach_epoch >= 150:
                    break

if __name__ == "__main__":
    main()