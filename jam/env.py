#coding: utf-8
from os import stat
import time
import numpy as onp
import jax
import jax.numpy as jnp
import jax.random as jrandom
from PIL import Image, ImageDraw, ImageOps
from pathlib import Path
from tqdm import tqdm
from enum import IntEnum, auto

class ObjectBase:
    def __init__(self, y, x, theta, dt, v_max, a_max):
        self.__y = y
        self.__x = x
        self.__theta = theta
        self.__v = 0.0

        self.__dt = dt
        self.__v_max = v_max
        self.__a_max = a_max

    # getter
    @property
    def y(self):
        return self.__y
    @property
    def x(self):
        return self.__x
    @property
    def v(self):
        return self.__v
    @property
    def theta(self):
        return self.__theta
    
    # setter
    @y.setter
    def y(self, val):
        self.__y = val
    @x.setter
    def x(self, val):
        self.__x = val
    @v.setter
    def v(self, val):
        self.__v = val
    @theta.setter
    def theta(self, val):
        self.__theta = val

    # y-position
    def calc_new_y(self, a, omega):
        return self.y + ObjectBase.__calc_dy(self.v, a, self.theta, omega, self.__dt, self.__v_max, self.__a_max)
    @staticmethod
    def __calc_dy(v, a, theta, omega, dt, v_max, a_max):
        dy = 0.0
        a = jnp.clip(a, -a_max, a_max)
        if v + a * dt > v_max:  # max speed limit
            dt_m = (v_max - v) / a
            dy = ObjectBase.__calc_dy(v, a, theta, omega, dt_m, v_max, a_max)
            dy += ObjectBase.__calc_dy(v_max, 0.0, theta + omega * dt_m, omega, dt - dt_m, v_max, a_max)
        elif v + a * dt < 0.0:  # min speed limit
            dt_m = v / a
            dy = ObjectBase.__calc_dy(v, a, theta, omega, dt_m, v_max, a_max)
        else:
            dy = ObjectBase.__calc_dy_impl(v, a, theta, omega, dt)
        return dy
    @staticmethod
    def __calc_dy_impl(v, a, theta, omega, dt):
        def integral_y(_v, _a, _theta, _omega, _dt):
            return _a / (_omega * _omega) * jnp.sin(_theta + _omega * _dt) - (_v + _a * _dt) / _omega * jnp.cos(_theta + _omega * _dt)
        if abs(omega) > 1E-2:
            # use omega only when not small, dut to numerical stability.
            dy = integral_y(v, a, theta, omega, dt) - integral_y(v, a, theta, omega, 0.0)
        else:
            dy = (v + 0.5 * a * dt) * dt * jnp.sin(theta)
        return dy

    # x-position
    def calc_new_x(self, a, omega):
        return self.x + ObjectBase.__calc_dx(self.v, a, self.theta, omega, self.__dt, self.__v_max, self.__a_max)
    @staticmethod
    def __calc_dx(v, a, theta, omega, dt, v_max, a_max):
        dx = 0.0
        a = jnp.clip(a, -a_max, a_max)
        if v + a * dt > v_max:  # max speed limit
            dt_m = (v_max - v) / a
            dx = ObjectBase.__calc_dx(v, a, theta, omega, dt_m, v_max, a_max)
            dx += ObjectBase.__calc_dx(v_max, 0.0, theta + omega * dt_m, omega, dt - dt_m, v_max, a_max)
        elif v + a * dt < 0.0:  # min speed limit
            dt_m = v / a
            dx = ObjectBase.__calc_dx(v, a, theta, omega, dt_m, v_max, a_max)
        else:
            dx = ObjectBase.__calc_dx_impl(v, a, theta, omega, dt)
        return dx
    @staticmethod
    def __calc_dx_impl(v, a, theta, omega, dt):
        def integral_x(_v, _a, _theta, _omega, _dt):
            return _a / (_omega * _omega) * jnp.cos(_theta + _omega * _dt) + (_v + _a * _dt) / _omega * jnp.sin(_theta + _omega * _dt)
        if abs(omega) > 1E-2:
            # use omega only when not small, dut to numerical stability.
            dx = integral_x(v, a, theta, omega, dt) - integral_x(v, a, theta, omega, 0.0)
        else:
            dx = (v + 0.5 * a * dt) * dt * jnp.cos(theta)
        return dx

    # speed
    def calc_new_v(self, a):
        return ObjectBase.__calc_new_v(self.v, a, self.__dt, self.__v_max, self.__a_max)
    @staticmethod
    def __calc_new_v(v, a, dt, v_max, a_max):
        a = jnp.clip(a, -a_max, a_max)
        return jnp.clip(v + a * dt, 0.0, v_max)

    # rotation
    def calc_new_theta(self, omega):
        return ObjectBase.__calc_new_theta(self.theta, omega, self.__dt)
    @staticmethod
    def __calc_new_theta(theta, omega, dt):
        new_theta = theta + omega * dt
        if new_theta < - jnp.pi:
            new_theta += 2.0 * jnp.pi
        elif new_theta > jnp.pi:
            new_theta -= 2.0 * jnp.pi
        return new_theta
    # update motion status
    def step_evolve(self, accel, omega, y_min, y_max, x_min, x_max, fin):
        new_theta = self.calc_new_theta(omega)
        new_y = self.calc_new_y(accel, omega)
        new_x = self.calc_new_x(accel, omega)
        new_v = self.calc_new_v(accel)

        self.theta = new_theta
        if (new_y > y_min) and (new_y < y_max) and (new_x > x_min) and (new_x < x_max) and (not fin):
            self.y = new_y
            self.x = new_x
            self.v = new_v
        else:
            self.v = 0.0

class Pedestrian(ObjectBase):
    radius_m = 0.5
    def __init__(self, y, x, theta, dt):
        v_max = 1.4
        a_max = v_max / 1.0
        super().__init__(y, x, theta, dt, v_max, a_max)

class PedestrianAgent(Pedestrian):
    def __init__(self, tgt_y, tgt_x, y, x, theta, dt):
        super().__init__(y, x, theta, dt)
        self.__tgt_y = tgt_y
        self.__tgt_x = tgt_x
    # getter
    @property
    def tgt_y(self):
        return self.__tgt_y
    @property
    def tgt_x(self):
        return self.__tgt_x
    
    def reached_goal(self):
        reached = False
        if  (abs(self.y - self.tgt_y) < Pedestrian.radius_m) and \
            (abs(self.x - self.tgt_x) < Pedestrian.radius_m):
            reached = True
        return reached
    
    def hit_with(self, other):
        hit = False
        if  (abs(self.y - other.tgt_y) < Pedestrian.radius_m) and \
            (abs(self.x - other.tgt_x) < Pedestrian.radius_m):
            hit = True
        return hit

class DelayRewardGen:
    def __init__(self, decay_rate, punish_max = 1.0):
        self.__reward = -1.0 * punish_max * (1.0 - decay_rate)
        self.__decay_rate = decay_rate
    def __call__(self):
        instant_reward = self.__reward
        self.__reward *= self.__decay_rate
        return instant_reward

class Environment:
    def __init__(self, rng, map_h, map_w, pcpt_h, pcpt_w, dt, n_ped_max):
        self.__rng, rng = jrandom.split(rng)
        self.__batch_size = 64
        self.__state_shape = (self.__batch_size, pcpt_h, pcpt_w, EnChannel.num)
        lr = 1E-2
        self.__policy = Policy(rng, map_h, map_w, nn_model(EnAction.num), self.__state_shape, lr)
        self.__n_ped_max = n_ped_max
        self.__map_h = map_h
        self.__map_w = map_w
        self.__dt = dt
        self.__agents = None
        self.__rewards = None
        self.__delay_reward_gens = None
        
    @property
    def n_ped_max(self):
        return self.__n_ped_max
    @property
    def map_h(self):
        return self.__map_h
    @property
    def map_w(self):
        return self.__map_w
    def get_agents(self):
        return self.__agents
    def get_rewards(self):
        return self.__rewards

    def __make_new_pedestrian(self, old_pedestrians):
        while 1:
            rng_y, rng_x, rng_theta, self.__rng = jrandom.split(self.__rng, 4)
            tgt_y, y = jrandom.uniform(rng_y, (2,), minval = Pedestrian.radius_m, maxval = self.map_h - Pedestrian.radius_m)
            tgt_x, x = jrandom.uniform(rng_x, (2,), minval = Pedestrian.radius_m, maxval = self.map_w - Pedestrian.radius_m)
            theta = jrandom.uniform(rng_theta, (1,), minval = 0.0, maxval = 2.0 * jnp.pi)[0]
            new_ped = PedestrianAgent(tgt_y, tgt_x, y, x, theta, self.__dt)

            isolated = True
            for old_ped in old_pedestrians:
                if  ((new_ped.tgt_y - old_ped.tgt_y) ** 2 + (new_ped.tgt_x - old_ped.tgt_x) ** 2 > Pedestrian.radius_m ** 2) and \
                    ((new_ped.y     - old_ped.y    ) ** 2 + (new_ped.x     - old_ped.x    ) ** 2 > Pedestrian.radius_m ** 2):
                    pass
                else:
                    isolated = False
                    break
            if isolated:
                break
        return new_ped
    
    def __make_init_state(self):
        _rng, self.__rng = jrandom.split(self.__rng, 2)
        n_ped = jrandom.randint(_rng, (1,), 1, self.n_ped_max + 1)

        agents = []
        delay_reward_gens = []
        for _ in range(int(n_ped)):
            new_ped = self.__make_new_pedestrian(agents)
            agents.append(new_ped)
            delay_reward_gens.append(DelayRewardGen(0.5 ** (1.0 / (100.0 / self.__dt))))

        return agents, delay_reward_gens

    def reset(self):
        self.__agents, self.__delay_reward_gens = self.__make_init_state()
        self.__rewards = []
        for _ in range(len(self.__agents)):
            self.__rewards.append(jnp.empty(0, dtype = jnp.float32))

    def __step_evolve(self, agent, action):
        accel, omega = action
        y_min = agent.radius_m
        y_max = self.map_h - agent.radius_m
        x_min = agent.radius_m
        x_max = self.map_w - agent.radius_m
        agent.step_evolve(accel, omega, y_min, y_max, x_min, x_max, agent.reached_goal())
        return agent
    
    def __calc_reward(self, agent_idx):
        reward = self.__delay_reward_gens[agent_idx]()
        other_agents = self.__agents[:agent_idx] + self.__agents[agent_idx + 1:]
        assert(len(other_agents) == len(self.__agents) - 1)

        for other_agent in other_agents:
            if self.__agents[agent_idx].hit_with(other_agent):
                reward += (-1.0)
        return reward
    
    def evolve(self, max_t):
        for _ in range(int(max_t / self.__dt)):
            for a in range(len(self.__agents)):
                if not self.__agents[a].reached_goal():
                    # action
                    obs_state = observe(self.__agents, a, self.__map_h, self.__map_w, self.__state_shape[1], self.__state_shape[2])
                    action = self.__policy(obs_state)
                    # update state
                    self.__agents[a] = self.__step_evolve(self.__agents[a], action)
                    # reward
                    reward = self.__calc_reward(a)
                    self.__rewards[a] = jnp.append(self.__rewards[a], reward)
            
            fin_all = True
            for a in range(len(self.__agents)):
                fin_all &= self.__agents[a].reached_goal()
            if fin_all:
                break
            
            yield self.__agents

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
        if 0: #random
            accel = 1.0 * 1.0 * jrandom.normal(rng_a)
            omega = 0.0 + jnp.pi * jrandom.normal(rng_o)
        elif 0: #best
            accel = 1.0
            tgt_theta = jnp.arctan2((agents[agent_idx].tgt_y - agents[agent_idx].y), (agents[agent_idx].tgt_x - agents[agent_idx].x))
            omega = (tgt_theta - agents[agent_idx].theta)
        else:
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
    occupy = Image.fromarray(onp.array(jnp.zeros((pcpt_h, pcpt_w, 3), dtype = jnp.uint8)))
    dr = ImageDraw.Draw(occupy)
    for a, agent in enumerate(agents):
        y = agent.y / map_h * pcpt_h
        x = agent.x / map_w * pcpt_w
        ry = agent.radius_m / map_h * pcpt_h 
        rx = agent.radius_m / map_w * pcpt_w
        
        py0 = jnp.clip(int((y - ry) + 0.5), 0, pcpt_h)
        py1 = jnp.clip(int((y + ry) + 0.5) + 1, 0, pcpt_h)
        px0 = jnp.clip(int((x - rx) + 0.5), 0, pcpt_w)
        px1 = jnp.clip(int((x + rx) + 0.5) + 1, 0, pcpt_w)
        dr.rectangle((px0, py0, px1, py1), fill = cols[a])

        ty = agent.tgt_y / map_h * pcpt_h 
        tx = agent.tgt_x / map_w * pcpt_w
        pty = jnp.clip(int(ty + 0.5), 0, pcpt_h)
        ptx = jnp.clip(int(tx + 0.5), 0, pcpt_w)
        lin_siz = 2
        dr.line((ptx - lin_siz, pty - lin_siz, ptx + lin_siz, pty + lin_siz), width = 1, fill = cols[a])
        dr.line((ptx - lin_siz, pty + lin_siz, ptx + lin_siz, pty - lin_siz), width = 1, fill = cols[a])
    dr.rectangle((0, 0, pcpt_w - 1, pcpt_h - 1), outline = WHITE)
    return occupy

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
def test():
    for seed in range(1):
        rng = jrandom.PRNGKey(seed)
        _rng, rng = jrandom.split(rng, 2)
        map_h = 50.0
        map_w = 50.0
        pcpt_h = 128
        pcpt_w = 128
        dt = 0.5
        n_ped_max = 4
        env = Environment(_rng, map_h, map_w, pcpt_h, pcpt_w, dt, n_ped_max)
        env.reset()

        dst_dir = Path("tmp/seed{}".format(seed))
        if not dst_dir.exists():
            dst_dir.mkdir(parents = True)
        log_path = dst_dir.joinpath("log.csv")
        log_writer = LogWriter(log_path)

        step = 0
        out_cnt = 0
        max_t = 100000.0
        for agents in tqdm(env.evolve(max_t)):
            out_infos = {}
            out_infos["step"] = step
            out_infos["t"] = step * dt
            for a, agent in enumerate(agents):
                agent_reward_log = env.get_rewards()[a]
                out_infos["tgt_y{}".format(a)] = agent.tgt_y
                out_infos["tgt_x{}".format(a)] = agent.tgt_x
                out_infos["y{}".format(a)] = agent.y
                out_infos["x{}".format(a)] = agent.x
                out_infos["v{}".format(a)] = agent.v
                out_infos["theta{}".format(a)] = agent.theta
                out_infos["r{}".format(a)] = agent_reward_log[-1]
                out_infos["total_r{}".format(a)] = agent_reward_log.sum()
                out_infos["len_of_r{}".format(a)] = agent_reward_log.size
                out_infos["reached_goal{}".format(a)] = agent.reached_goal()
            log_writer.write(out_infos)
            
            if step % int(1.0 / dt) == 0:
                dst_path = dst_dir.joinpath("png", "{}.png".format(out_cnt))
                if not dst_path.exists():
                    if 1:
                        pcpt_h = 128
                        pcpt_w = 128
                        img = ImageOps.flip(make_state_img(agents, map_h, map_w, pcpt_h, pcpt_w))
                        img = get_concat_h(img, Image.fromarray((255 * ((observe(agents, 0, map_h, map_w, pcpt_h, pcpt_w)[::-1,:,EnChannel.occupy] + 1.0) / 2)).astype(jnp.uint8)))
                        img = get_concat_h(img, Image.fromarray((255 * ((observe(agents, 0, map_h, map_w, pcpt_h, pcpt_w)[::-1,:,EnChannel.vy    ] + 1.5) / 3)).astype(jnp.uint8)))
                        img = get_concat_h(img, Image.fromarray((255 * ((observe(agents, 0, map_h, map_w, pcpt_h, pcpt_w)[::-1,:,EnChannel.vx    ] + 1.5) / 3)).astype(jnp.uint8)))

                        if not dst_path.parent.exists():
                            dst_path.parent.mkdir(parents = True)
                        dr = ImageDraw.Draw(img)
                        dr.text((0,0), "{}".format(step * dt), fill = WHITE)
                        if not dst_path.parent.exists():
                            dst_path.parent.mkdir(parents = True)
                        img.save(dst_path)
                out_cnt += 1

            step += 1
        print(step * dt)
        for reward_vec in env.get_rewards():
            print(reward_vec.sum(), end = ",")
        print()

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
import hydra
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

class TrainerBase:
    def __init__(self):
        pass

class Trainer(TrainerBase):
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

def main():
    test()
    exit()
    m = Trainer(jax.random.PRNGKey(0))
    m.learn()

if __name__ == "__main__":
    main()