#coding: utf-8
from os import stat
import time
import numpy as onp
import jax
import jax.numpy as jnp
import jax.random as jrandom
from pathlib import Path
from tqdm import tqdm
from collections import namedtuple, deque
import subprocess
import hydra

from net import SharedNetwork
from objects import PedestrianObject
from delay import DelayGen
from common import EnAction, EnChannel
from observer import observe
from log import LogWriter
from vis import make_all_state_img

Experience = namedtuple("Experience",
                        [
                            "observation",
                            "action",
                            "reward",
                            "next_state",
                            "finished",
                            "next_finished",
                        ]
                        )

class Environment:
    def __init__(self, rng, batch_size, map_h, map_w, pcpt_h, pcpt_w, max_t, dt, half_decay_dt, n_ped_max):
        self.__rng, rng = jrandom.split(rng)
        self.__batch_size = batch_size
        self.__state_shape = (self.__batch_size, pcpt_h, pcpt_w, EnChannel.num)
        self.__n_ped_max = n_ped_max
        self.__map_h = map_h
        self.__map_w = map_w
        self.__max_t = max_t
        self.__dt = dt
        self.__objects = None
        self.__gamma = 0.9999#0.5 ** (1.0 / (half_decay_dt / self.dt))
        self.debug = 0

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
    def policy(self):
        return self.__policy
    @property
    def gamma(self):
        return self.__gamma
    @property
    def state_shape(self):
        return self.__state_shape
    def get_obj_num(self):
        return len(self.__objects)

    def __make_new_pedestrian(self, existing_objects):
        _rng, self.__rng = jrandom.split(self.__rng, 2)
        while 1:
            rng_y, rng_x, rng_theta, _rng = jrandom.split(_rng, 4)
            tgt_y, y = jrandom.uniform(rng_y, (2,), minval = 0.0, maxval = self.map_h)
            tgt_x, x = jrandom.uniform(rng_x, (2,), minval = 0.0, maxval = self.map_w)
            theta = jrandom.uniform(rng_theta, (1,), minval = 0.0, maxval = 2.0 * jnp.pi)[0]
            new_ped = PedestrianObject(tgt_y, tgt_x, y, x, theta, self.dt)

            isolated = False
            if  (new_ped.y >        0.0 + new_ped.radius_m) and \
                (new_ped.y < self.map_h - new_ped.radius_m) and \
                (new_ped.x >        0.0 + new_ped.radius_m) and \
                (new_ped.x < self.map_w - new_ped.radius_m) and \
                (new_ped.tgt_y >        0.0 + new_ped.radius_m) and \
                (new_ped.tgt_y < self.map_h - new_ped.radius_m) and \
                (new_ped.tgt_x >        0.0 + new_ped.radius_m) and \
                (new_ped.tgt_x < self.map_w - new_ped.radius_m):
                if not new_ped.reached_goal():
                    far_from_all_others = True
                    for existing_object in existing_objects:
                        if new_ped.hit_with(existing_object):
                            far_from_all_others = False
                            break
                    isolated = far_from_all_others

            if isolated:
                break
        return new_ped
    
    def __make_init_state(self):
        _rng, self.__rng = jrandom.split(self.__rng, 2)
        n_ped = self.n_ped_max#jrandom.randint(_rng, (1,), 1, self.n_ped_max + 1)

        objects = []
        for _ in range(int(n_ped)):
            new_ped = self.__make_new_pedestrian(objects)
            objects.append(new_ped)

        return objects

    def reset(self):
        self.__objects = self.__make_init_state()

        observation = []
        for obj_idx, obj in enumerate(self.__objects):
            next_state = observe(self.__objects, obj_idx, self.map_h, self.map_w, self.__state_shape[1], self.__state_shape[2])
            observation.append(next_state)
        return observation

    def __step_evolve(self, object, action):
        accel, omega = action
        y_min = object.radius_m
        y_max = self.map_h - object.radius_m
        x_min = object.radius_m
        x_max = self.map_w - object.radius_m
        object.step(accel, omega, y_min, y_max, x_min, x_max, object.reached_goal() or object.hit_with_wall(self.map_h, self.map_w))
        return object
    
    def __calc_reward(self, obj_idx, action):
        reward = 0.0
        own = self.__objects[obj_idx]

        # delay punishment
        reward = reward + (- 0.01)# * (1.0 - self.__gamma)#self.__delay_reward()
        
        # hit
        other_objects = self.__objects[:obj_idx] + self.__objects[obj_idx + 1:]
        assert(len(other_objects) == len(self.__objects) - 1)
        for other_object in other_objects:
            if own.hit_with(other_object):
                reward += (-1.0)
        
        # approach
        #remain_distance = ((own.y - own.tgt_y) ** 2 + (own.x - own.tgt_x) ** 2) ** 0.5
        #max_distance = (self.map_h ** 2 + self.map_w ** 2) ** 0.5
        #remain_rate = remain_distance / max_distance
        #reward = reward + (- 1.0) * remain_rate
        
        # reach
        if own.reached_goal():
            reward = reward + (+1.0)
        
        # hit with wall
        #if own.hit_with_wall(self.map_h, self.map_w):
        #    reward += (-1.0)

        # punish extreme action
        #reward = reward - 0.005 * (action * action).mean() * (1.0 - self.__gamma)

        return jnp.array(reward)
    
    def step(self, action):

        observation = []
        done = []
        reward = []
        done_already = []
        info = {}
        # play
        for obj_idx in range(len(self.__objects)):
            done1 = self.__objects[obj_idx].reached_goal()# or self.__objects[a].hit_with_wall(self.map_h, self.map_w)
            '''
            out_cnt = 0
            while 1:
                out_cnt += 1
                out_path = Path("/home/isgsktyktt/work/im/{}/{}.png".format(obj_idx, out_cnt))
                if not out_path.exists():
                    if not out_path.parent.exists():
                        out_path.parent.mkdir(parents = True)
                    break
            img = make_all_state_img(self.__objects, obj_idx, self.map_h, self.map_w, pcpt_h = self.__state_shape[1], pcpt_w = self.__state_shape[2])
            w, h = img.size
            rate = 8
            img = img.resize((w*rate, h*rate))
            img.save(out_path)
            pass
            '''

            # state transition
            if not done1:
                self.__objects[obj_idx] = self.__step_evolve(self.__objects[obj_idx], action[obj_idx])
            else:
                self.__objects[obj_idx].stop()
            
            done_already.append(done1)
            
        # evaluation
        for obj_idx, obj in enumerate(self.__objects):
            next_state = observe(self.__objects, obj_idx, self.map_h, self.map_w, self.__state_shape[1], self.__state_shape[2])
            if done_already[obj_idx]:
                r = 0.0
            else:
                r = self.__calc_reward(obj_idx, action)
            d = obj.reached_goal()# or obj.hit_with_wall(self.map_h, self.map_w)

            observation.append(next_state)
            reward.append(r)
            done.append(d)

            info["ini_x{}".format(obj_idx)] = obj.ini_x
            info["ini_y{}".format(obj_idx)] = obj.ini_y
            info["tgt_x{}".format(obj_idx)] = obj.tgt_x
            info["tgt_y{}".format(obj_idx)] = obj.tgt_y
            info["x{}".format(obj_idx)] = obj.x
            info["y{}".format(obj_idx)] = obj.y
            info["v{}".format(obj_idx)] = obj.v
            info["theta{}".format(obj_idx)] = obj.theta
            
        return observation, reward, done, info
    def action_abs_max(self, obj_idx):
        return self.__objects[obj_idx].action_abs_max

class Agent:
    def __init__(self, cfg_net, rng, batch_size, init_weight_path, pcpt_h, pcpt_w) -> None:
        self.shared_nn = SharedNetwork(cfg_net, rng, init_weight_path, batch_size, pcpt_h, pcpt_w)
    def get_action(self, state, action_abs_max, explore):
        action, means, sigs = self.shared_nn.decide_action(state, explore)
        means = means.flatten()
        sigs = sigs.flatten()
        action = (action * action_abs_max).flatten()   # single object size
        return action, means, sigs

class Trainer:

    def __init__(self, cfg, seed):
        self.__cfg = cfg.train
        rng = jrandom.PRNGKey(seed)
        self.__rng, rng_e, rng_a = jrandom.split(rng, 3)
        self.__batch_size = 256
        map_h = 10.0
        map_w = 10.0
        pcpt_h = 32
        pcpt_w = 32
        max_t = 5000.0
        dt = 0.5 * 2
        n_ped_max = 1
        half_decay_dt = 10.0
        init_weight_path = None#"/home/isgsktyktt/work/init_param.bin"
        self.__buf_max = int(1E5)#self.__batch_size * self.__batch_size
        self.__experiences = deque(maxlen = self.__buf_max)

        self.__env = Environment(rng_e, self.__batch_size, map_h, map_w, pcpt_h, pcpt_w, max_t, dt, half_decay_dt, n_ped_max)
        self.__agent = Agent(cfg.net, rng_a, self.__batch_size, init_weight_path, pcpt_h, pcpt_w)
    def __evolve(self, explore):
        observation = self.__env.reset()
        obj_num = self.__env.get_obj_num()
        
        observation_old = observation
        done_old = None
        max_step = int(self.__env.max_t / self.__env.dt)
        for _ in range(max_step):
            # decide action
            action = []
            out_info = {}
            for obj_idx in range(obj_num):
                act, means, sigs = self.__agent.get_action(observation[obj_idx], self.__env.action_abs_max(obj_idx), explore)
                action.append(act)
                for a, (act1, mean, sigma) in enumerate(zip(act, means, sigs)):
                    out_info["obj{}_action{}".format(obj_idx, a)] = act1
                    out_info["obj{}_mean{}".format(obj_idx, a)] = mean
                    out_info["obj{}_sigma{}".format(obj_idx, a)] = sigma
                out_info["Q{}".format(obj_idx)] = float(self.__agent.shared_nn.apply_Q_smaller(observation[obj_idx], act.reshape((1, -1))))
            # state transition
            observation, reward, done, info = self.__env.step(action)
            out_info.update(info)
            
            if done_old is None:
                done_old = [False for _ in range(obj_num)]
            
            new_es = []
            for obj_idx in range(obj_num):
                if not done_old[obj_idx]:
                    new_e = Experience( observation_old[obj_idx],
                                        action[obj_idx].flatten(),
                                        reward[obj_idx],
                                        observation[obj_idx],
                                        done_old[obj_idx],
                                        done[obj_idx],
                                        )
                    new_es.append(new_e)
                out_info["reward{}".format(obj_idx)] = float(reward[obj_idx])
                out_info["done{}".format(obj_idx)] = done_old[obj_idx]
            observation_old = observation
            done_old = done
            
            yield out_info, new_es
            
            if jnp.array(done).all():
                break
    def learn_episode(self, verbose = True):
        episode_num_per_unit = 1
        learn_num_per_unit = 16
        
        dst_base_dir = Path(self.__cfg.dst_dir_path)
        log_writer = None
        all_log_writer = LogWriter(dst_base_dir.joinpath("learn.csv"))
        for trial in range(self.__cfg.episode_unit_num):
            if (trial + 1) % 64 == 0:
                explore = False
                weight_path = dst_base_dir.joinpath("weight", "param{}.bin".format(trial))
                if not weight_path.parent.exists():
                    weight_path.parent.mkdir(parents = True)
                self.__agent.shared_nn.save(weight_path)
            else:
                explore = True
            for episode_cnt in range(episode_num_per_unit):
                log_path = dst_base_dir.joinpath("play", "{}_{}.csv".format(trial, episode_cnt))
                if log_writer is not None:
                    del log_writer
                log_writer = LogWriter(log_path)
            
                if verbose:
                    loop_fun = tqdm
                else:
                    loop_fun = lambda x : x

                for step, (info, new_es) in loop_fun(enumerate(self.__evolve(explore))):
                    if explore:
                        for new_e in new_es:
                            self.__experiences.append(new_e)

                    out_infos = {}
                    out_infos["episode"] = episode_cnt
                    out_infos["step"] = step
                    out_infos["t"] = step * self.__env.dt
                    out_infos.update(info)
                    out_infos["explore"] = explore
                    log_writer.write(out_infos)
                    
                # after episode

            if len(self.__experiences) < self.__batch_size:
                continue
            # after episode unit
            learn_cnt_per_unit = 0
            state_shape = self.__env.state_shape
            s = jnp.zeros(state_shape, dtype = jnp.float32)
            a = jnp.zeros((state_shape[0], EnAction.num), dtype = jnp.float32)
            r = jnp.zeros((state_shape[0], 1), dtype = jnp.float32)
            n_s = jnp.zeros(state_shape, dtype = jnp.float32)
            n_fin = jnp.zeros((state_shape[0], 1), dtype = jnp.float32)
            gamma = self.__env.gamma
            val = 0
            total_loss_q = []
            total_loss_pi = []
            while 1:
                self.__rng, rng = jrandom.split(self.__rng)
                e_i = int(jrandom.randint(rng, (1,), 0, len(self.__experiences)))
                e = self.__experiences[e_i]
                if not e.finished:
                    s = s.at[val,:].set(e.observation[0])
                    a = a.at[val,:].set(e.action)
                    r = r.at[val].set(float(e.reward))
                    n_s = n_s.at[val,:].set(e.next_state[0])
                    n_fin = n_fin.at[val,:].set(float(e.next_finished))
                    val += 1
                    if val >= state_shape[0]:
                        q_learn_cnt, p_learn_cnt, temperature, loss_val_qs, loss_val_pi, loss_balances = self.__agent.shared_nn.update(gamma, s, a, r, n_s, n_fin)
                        all_info = {}
                        all_info["trial"] = int(trial)
                        all_info["episode_num_per_unit"] = int(episode_num_per_unit)
                        all_info["episode"] = int(episode_cnt)
                        all_info["q_learn_cnt"] = int(q_learn_cnt)
                        #all_info["p_learn_cnt"] = int(p_learn_cnt)
                        all_info["temperature"] = float(temperature)
                        for _i, loss_val_q in enumerate(loss_val_qs):
                            all_info["loss_val_q{}".format(_i)] = float(loss_val_q)
                        all_info["loss_val_pi"] = float(loss_val_pi)
                        #for _i, loss_balance in enumerate(loss_balances):
                        #    all_info["loss_balance{}".format(_i)] = float(loss_balance)
                        all_log_writer.write(all_info)
                        if verbose:
                            for value in all_info.values():
                                if isinstance(value, float):
                                    print("{:.3f}".format(value), end = ",")
                                else:
                                    print(value, end = ",")
                            print()
                        total_loss_q.append(loss_val_q)
                        total_loss_pi.append(loss_val_pi)
                        val = 0
                        learn_cnt_per_unit += 1
                        if (learn_cnt_per_unit >= min(learn_num_per_unit, len(self.__experiences) // self.__batch_size)):
                            break

            #episode_num_per_unit = min(episode_num_per_unit + 1, state_shape[0])
            
@hydra.main(config_path = ".", config_name = "main.yaml")
def main(cfg):
    seed = 0
    trainer = Trainer(cfg, seed)
    trainer.learn_episode()

if __name__ == "__main__":
    main()
