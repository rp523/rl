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

    def __make_new_pedestrian(self, old_pedestrians):
        _rng, self.__rng = jrandom.split(self.__rng, 2)
        while 1:
            rng_y, rng_x, rng_theta, _rng = jrandom.split(_rng, 4)
            tgt_y, y = jrandom.uniform(rng_y, (2,), minval = 0.0, maxval = self.map_h)
            tgt_x, x = jrandom.uniform(rng_x, (2,), minval = 0.0, maxval = self.map_w)
            theta = jrandom.uniform(rng_theta, (1,), minval = 0.0, maxval = 2.0 * jnp.pi)[0]
            new_ped = PedestrianObject(tgt_y, tgt_x, y, x, theta, self.dt)

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
                    if  ((new_ped.tgt_y - old_ped.tgt_y) ** 2 + (new_ped.tgt_x - old_ped.tgt_x) ** 2 > (2 * (new_ped.radius_m + old_ped.radius_m)) ** 2) and \
                        ((new_ped.y     - old_ped.y    ) ** 2 + (new_ped.x     - old_ped.x    ) ** 2 > (2 * (new_ped.radius_m + old_ped.radius_m)) ** 2):
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
        #reward = reward + (- 0.01)# * (1.0 - self.__gamma)#self.__delay_reward()
        
        # hit
        #other_objects = self.__objects[:obj_idx] + self.__objects[obj_idx + 1:]
        #assert(len(other_objects) == len(self.__objects) - 1)
        #for other_object in other_objects:
        #    break
        #    if own.hit_with(other_object):
        #        reward += (-1.0)
        
        # approach
        remain_distance = ((own.y - own.tgt_y) ** 2 + (own.x - own.tgt_x) ** 2) ** 0.5
        max_distance = (self.map_h ** 2 + self.map_w ** 2) ** 0.5
        remain_rate = remain_distance / max_distance
        reward = reward + (- 1.0) * remain_rate
        
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
            img = make_all_state_img(self.__objects, self.map_h, self.map_w, pcpt_h = self.__state_shape[1], pcpt_w = self.__state_shape[2])
            w, h = img.size
            rate = 8
            img = img.resize((w*rate, h*rate))
            #img.show();exit()
            img.save("/home/isgsktyktt/work/im/{}.png".format(self.debug))#;exit()
            self.debug += 1
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

            info["tgt_x{}".format(obj_idx)] = obj.tgt_x
            info["tgt_y{}".format(obj_idx)] = obj.tgt_y
            info["x{}".format(obj_idx)] = obj.x
            info["y{}".format(obj_idx)] = obj.y
            info["v{}".format(obj_idx)] = obj.v
            info["theta{}".format(obj_idx)] = obj.theta
            
        return observation, reward, done, info

class Agent:
    def __init__(self, cfg_net, rng, batch_size, init_weight_path, pcpt_h, pcpt_w) -> None:
        self.shared_nn = SharedNetwork(cfg_net, rng, init_weight_path, batch_size, pcpt_h, pcpt_w)
    def get_action(self, state, explore):
        action, means, sigs = self.shared_nn.decide_action(state, explore)
        action = action.flatten()   # single object size
        return action, means, sigs

class Trainer:

    def __init__(self, cfg, seed):
        self.__cfg = cfg.train
        rng = jrandom.PRNGKey(seed)
        self.__rng, rng_e, rng_a = jrandom.split(rng, 3)
        self.__batch_size = 256
        map_h = 10.0
        map_w = 10.0
        pcpt_h = 16
        pcpt_w = 16
        max_t = 10000.0
        dt = 0.5 * 4
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
                act, means, sigs = self.__agent.get_action(observation[obj_idx], explore)
                action.append(act)
                for a, (mean, sigma) in enumerate(zip(means[obj_idx], sigs[obj_idx])):
                    out_info["obj{}_mean{}".format(obj_idx, a)] = mean
                    out_info["obj{}_sigma{}".format(obj_idx, a)] = sigma
            # state transition
            observation, reward, done, info = self.__env.step(action)
            out_info.update(info)
            
            if done_old is None:
                done_old = [False for _ in range(obj_num)]
            
            new_es = []
            for obj_idx in range(obj_num):
                new_e = Experience( observation_old[obj_idx],
                                    action[obj_idx].flatten(),
                                    reward[obj_idx],
                                    observation[obj_idx],
                                    done_old[obj_idx],
                                    done[obj_idx],
                                    )
                new_es.append(new_e)
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
                    obj_num = self.__env.get_obj_num()
                    for obj_idx in range(obj_num):
                        experience = new_es[obj_idx]
                        out_infos["accel{}".format(obj_idx)] = float(experience.action[0])
                        out_infos["omega{}".format(obj_idx)] = float(experience.action[1])
                        out_infos["Q{}".format(obj_idx)] = float(self.__agent.shared_nn.apply_Q_smaller(experience.observation, experience.action.reshape((1, -1))))
                        out_infos["reward{}".format(obj_idx)] = float(experience.reward)
                    out_infos["explore"] = explore
                    log_writer.write(out_infos)
                    
                # after episode
                exit()

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
            weight_path = dst_base_dir.joinpath("weight", "param{}.bin".format(trial))
            if not weight_path.parent.exists():
                weight_path.parent.mkdir(parents = True)
            self.__agent.shared_nn.save(weight_path)

            #episode_num_per_unit = min(episode_num_per_unit + 1, state_shape[0])
            
@hydra.main(config_path = ".", config_name = "main.yaml")
def main(cfg):
    seed = 0
    trainer = Trainer(cfg, seed)
    trainer.learn_episode()

if __name__ == "__main__":
    main()
