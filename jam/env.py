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
from agent import PedestrianAgent
from delay import DelayGen
from common import *
from observer import observe
from log import LogWriter
from vis import make_all_state_img

Experience = namedtuple("Experience",
                        [
                            "state",
                            "action",
                            "reward",
                            "next_state",
                            "finished",
                            "accel_mean",
                            "accel_sigma",
                            "omega_mean",
                            "omega_sigma",
                        ]
                        )

class Environment:
    def __init__(self, cfg, rng, init_weight_path, batch_size, map_h, map_w, pcpt_h, pcpt_w, max_t, dt, half_decay_dt, n_ped_max):
        self.__rng, rng = jrandom.split(rng)
        self.__batch_size = batch_size
        self.__state_shape = (self.__batch_size, pcpt_h, pcpt_w, EnChannel.num)
        self.__shared_nn = SharedNetwork(cfg.net, rng, init_weight_path, self.__batch_size, pcpt_h, pcpt_w)
        self.__n_ped_max = n_ped_max
        self.__map_h = map_h
        self.__map_w = map_w
        self.__max_t = max_t
        self.__dt = dt
        self.__agents = None
        self.__experiences = deque(maxlen = batch_size ** 2)
        self.__gamma = 0.99#0.5 ** (1.0 / (half_decay_dt / self.dt))

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

        agents = []
        for _ in range(int(n_ped)):
            new_ped = self.__make_new_pedestrian(agents)
            agents.append(new_ped)

        return agents

    def reset(self):
        self.__agents = self.__make_init_state()

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
        max_step = int(self.max_t / self.dt)
        own = self.__agents[agent_idx]

        # delay punishment
        reward += (- 0.1) / max_step#self.__delay_reward()
        # hit
        other_agents = self.__agents[:agent_idx] + self.__agents[agent_idx + 1:]
        assert(len(other_agents) == len(self.__agents) - 1)
        for other_agent in other_agents:
            break
            if own.hit_with(other_agent):
                reward += (-1.0)
        # approach
        remain_distance = jnp.sqrt((own.y - own.tgt_y) ** 2 + (own.x - own.tgt_x) ** 2)
        max_distance = jnp.sqrt(self.__map_h ** 2 + self.__map_w ** 2)
        approach_rate = (max_distance - remain_distance) / max_distance
        reward += (+ 0.5) * approach_rate / max_step
        # reach
        if own.reached_goal():
            reward += 0.5
        return reward
    
    def evolve(self):
        for _ in range(int(self.max_t / self.dt)):
            rec = []
            for a in range(len(self.__agents)):
                fin = self.__agents[a].reached_goal()
                state = observe(self.__agents, a, self.map_h, self.map_w, self.__state_shape[1], self.__state_shape[2])
                #make_all_state_img(self.__agents, self.map_h, self.map_w, pcpt_h = self.__state_shape[1], pcpt_w = self.__state_shape[2]).show();exit()
                
                action, a_mean, a_sig, o_mean, o_sig = self.__shared_nn.decide_action(state)
                action = action.flatten()   # single agent size

                if not fin:
                    self.__agents[a] = self.__step_evolve(self.__agents[a], action)
                    reward = self.__calc_reward(a)
                else:
                    self.__agents[a].stop()
                    reward = 0.0
                
                rec.append((state, action, reward, fin, a_mean, a_sig, o_mean, o_sig))
            for a in range(len(self.__agents)):
                next_state = observe(self.__agents, a, self.__map_h, self.__map_w, self.__state_shape[1], self.__state_shape[2])
                #next_action = self.__shared_nn.decide_action(next_state)
                #next_action = next_action.flatten()   # single agent size

                #self.__agents[a].reserved_action = next_action
                state, action, reward, fin, a_mean, a_sig, o_mean, o_sig = rec[a]
                experience = Experience(state, action.flatten(), reward, next_state, fin, a_mean, a_sig, o_mean, o_sig)
                self.__experiences.append(experience)

            fin_all = True
            for a in range(len(self.__agents)):
                fin_all &= self.__agents[a].reached_goal()
            yield fin_all, self.__agents
            
            if fin_all:
                break
            
class Trainer:
    def __init__(self, cfg, seed):
        self.__cfg = cfg
        rng = jrandom.PRNGKey(seed)
        self.__rng, rng = jrandom.split(rng)
        batch_size = 256
        map_h = 10.0
        map_w = 10.0
        pcpt_h = 32
        pcpt_w = 32
        max_t = 100.0
        dt = 0.5
        n_ped_max = 1
        half_decay_dt = 10.0
        init_weight_path = None#"/home/isgsktyktt/work/init_param.bin"
        self.__buf_max = batch_size * batch_size

        self.__env = Environment(cfg, rng, init_weight_path, batch_size, map_h, map_w, pcpt_h, pcpt_w, max_t, dt, half_decay_dt, n_ped_max)
    def learn_episode(self, verbose = True):
        episode_unit_num = 1000000
        episode_num_per_unit = 1
        learn_num_per_unit = 1
        
        dst_base_dir = Path("/home/isgsktyktt/work/tmp")
        log_writer = None
        all_log_writer = LogWriter(dst_base_dir.joinpath("learn.csv"))
        for trial in range(episode_unit_num):
            total_rewards = []
            for episode in range(episode_num_per_unit):
                log_path = dst_base_dir.joinpath("play", "{}_{}.csv".format(trial, episode))
                if log_writer is not None:
                    del log_writer
                log_writer = LogWriter(log_path)
                self.__env.reset()
            
                total_reward = [0.0] * len(self.__env.get_agents())
                if verbose:
                    loop_fun = tqdm
                else:
                    loop_fun = lambda x : x
                for step, (fin_all, agents) in loop_fun(enumerate(self.__env.evolve())):
                    out_infos = {}
                    out_infos["episode"] = episode
                    out_infos["step"] = step
                    out_infos["t"] = step * self.__env.dt
                    for agent_idx, agent in enumerate(agents):
                        experience = self.__env.experiences[-len(agents) + agent_idx]
                        out_infos["tgt_y{}".format(agent_idx)] = float(agent.tgt_y)
                        out_infos["tgt_x{}".format(agent_idx)] = float(agent.tgt_x)
                        out_infos["y{}".format(agent_idx)] = float(agent.y)
                        out_infos["x{}".format(agent_idx)] = float(agent.x)
                        out_infos["v{}".format(agent_idx)] = float(agent.v)
                        out_infos["theta{}".format(agent_idx)] = float(agent.theta)
                        out_infos["accel{}".format(agent_idx)] = float(experience.action[0])
                        out_infos["accel_mean{}".format(agent_idx)] = float(experience.accel_mean)
                        out_infos["accel_sigma{}".format(agent_idx)] = float(experience.accel_sigma)
                        out_infos["omega{}".format(agent_idx)] = float(experience.action[1])
                        out_infos["omega_mean{}".format(agent_idx)] = float(experience.omega_mean)
                        out_infos["omega_sigma{}".format(agent_idx)] = float(experience.omega_sigma)
                        out_infos["finished{}".format(agent_idx)] = experience.finished
                        total_reward[agent_idx] += experience.reward
                        out_infos["reward{}".format(agent_idx)] = float(experience.reward)
                        out_infos["total_reward{}".format(agent_idx)] = float(total_reward[agent_idx])
                    log_writer.write(out_infos)
                    
                # after episode
                for t in total_reward:
                    total_rewards.append(float(t))

            #if len(self.__env.experiences) < self.__buf_max:
            #    continue
            # after episode unit
            learn_cnt_per_unit = 0
            total_reward_mean = float(jnp.array(total_rewards).mean())
            state_shape = self.__env.state_shape
            s = jnp.zeros(state_shape, dtype = jnp.float32)
            a = jnp.zeros((state_shape[0], EnAction.num), dtype = jnp.float32)
            r = jnp.zeros((state_shape[0], 1), dtype = jnp.float32)
            n_s = jnp.zeros(state_shape, dtype = jnp.float32)
            gamma = self.__env.gamma
            val = 0
            total_loss_q = []
            total_loss_pi = []
            while 1:
                self.__rng, rng = jrandom.split(self.__rng)
                e_i = int(jrandom.randint(rng, (1,), 0, len(self.__env.experiences)))
                e = self.__env.experiences[e_i]
                if not e.finished:
                    s = s.at[val,:].set(e.state[0])
                    a = a.at[val,:].set(e.action)
                    r = r.at[val].set(e.reward.flatten())
                    n_s = n_s.at[val,:].set(e.next_state[0])
                    val += 1
                    if val >= state_shape[0]:
                        q_learn_cnt, p_learn_cnt, temperature, loss_val_qs, loss_val_pi, loss_balances = self.__env.shared_nn.update(gamma, s, a, r, n_s)
                        all_info = {}
                        all_info["trial"] = int(trial)
                        all_info["episode_num_per_unit"] = int(episode_num_per_unit)
                        all_info["episode"] = int(episode)
                        all_info["q_learn_cnt"] = int(q_learn_cnt)
                        #all_info["p_learn_cnt"] = int(p_learn_cnt)
                        all_info["temperature"] = float(temperature)
                        all_info["total_reward_mean"] = float(total_reward_mean)
                        for _i, loss_val_q in enumerate(loss_val_qs):
                            all_info["loss_val_q{}".format(_i)] = float(loss_val_q)
                        all_info["loss_val_pi"] = float(loss_val_pi)
                        for _i, loss_balance in enumerate(loss_balances):
                            all_info["loss_balance{}".format(_i)] = float(loss_balance)
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
                        if (learn_cnt_per_unit >= learn_num_per_unit):
                            break
            weight_path = dst_base_dir.joinpath("weight", "param{}.bin".format(trial))
            if not weight_path.parent.exists():
                weight_path.parent.mkdir(parents = True)
            self.__env.shared_nn.save(weight_path)

            #episode_num_per_unit = min(episode_num_per_unit + 1, state_shape[0])
            
@hydra.main(config_path = ".", config_name = "main.yaml")
def main(cfg):
    seed = 0
    trainer = Trainer(cfg, seed)
    trainer.learn_episode()

if __name__ == "__main__":
    main()
