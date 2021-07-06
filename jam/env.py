#coding: utf-8
from os import stat
import time
import numpy as onp
import jax
import jax.numpy as jnp
import jax.random as jrandom
from pathlib import Path
from tqdm import tqdm
from collections import namedtuple
import subprocess

from net import SharedNetwork
from agent import PedestrianAgent
from delay import DelayGen
from common import *
from observer import observe
from log import LogWriter

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
        self.__shared_nn = SharedNetwork(rng, init_weight_path, self.__batch_size, pcpt_h, pcpt_w)
        self.__n_ped_max = n_ped_max
        self.__map_h = map_h
        self.__map_w = map_w
        self.__max_t = max_t
        self.__dt = dt
        self.__agents = None
        self.__experiences = []
        self.__gamma = 0.5 ** (1.0 / (half_decay_dt / self.dt))
        self.__r_decay = 0.5 ** (1.0 / (self.max_t / self.dt))
        self.__delay_reward = DelayGen(decay_rate = self.__r_decay, accum_max = - 0.1)
        self.__approach_reward = DelayGen(decay_rate = self.__r_decay, accum_max = 0.1)

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
        self.__delay_reward.reset()
        self.__approach_reward.reset()
        self.__experiences.clear()

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

        # delay punishment
        reward += self.__delay_reward()
        # hit
        other_agents = self.__agents[:agent_idx] + self.__agents[agent_idx + 1:]
        assert(len(other_agents) == len(self.__agents) - 1)
        for other_agent in other_agents:
            break
            if own.hit_with(other_agent):
                reward += (-1.0)
        # approach
        approach_rate = (jnp.sqrt((own.tgt_y - own.init_y) ** 2 + (own.tgt_x - own.init_x) ** 2) - jnp.sqrt((own.y - own.tgt_y) ** 2 + (own.x - own.tgt_x) ** 2)) / jnp.sqrt(self.__map_h ** 2 + self.__map_w ** 2)
        reward += self.__approach_reward() * approach_rate
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
                experience = Experience(state, action.flatten(), reward, next_state, next_action.flatten(), fin)
                self.__experiences.append(experience)
            self.__delay_reward.step()
            self.__approach_reward.step()

            fin_all = True
            for a in range(len(self.__agents)):
                fin_all &= self.__agents[a].reached_goal()
            yield fin_all, self.__agents
            
            if fin_all:
                break
            
class Trainer:
    def __init__(self, seed = 0):
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
        init_weight_path = None#r"/home/isgsktyktt/work/tmp/param0.bin"

        self.__env = Environment(rng, init_weight_path, batch_size, map_h, map_w, pcpt_h, pcpt_w, max_t, dt, half_decay_dt, n_ped_max)
    def learn_episode(self, verbose = True):
        dst_base_dir = Path("tmp")
        log_writer = None
        all_log_writer = LogWriter(dst_base_dir.joinpath("loss.csv"))
        total_log = False
        for trial in range(1000):
            self.__env.reset()

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
                    out_infos["tgt_y{}".format(agent_idx)] = float(agent.tgt_y)
                    out_infos["tgt_x{}".format(agent_idx)] = float(agent.tgt_x)
                    out_infos["y{}".format(agent_idx)] = float(agent.y)
                    out_infos["x{}".format(agent_idx)] = float(agent.x)
                    out_infos["v{}".format(agent_idx)] = float(agent.v)
                    out_infos["theta{}".format(agent_idx)] = float(agent.theta)
                    out_infos["accel{}".format(agent_idx)] = float(experience.action[0])
                    out_infos["omega{}".format(agent_idx)] = float(experience.action[1])
                    out_infos["reward{}".format(agent_idx)] = float(experience.reward)
                    out_infos["finished{}".format(agent_idx)] = experience.finished
                    total_reward[agent_idx] += experience.reward
                log_writer.write(out_infos)
                
                if (step % int(1.0 / self.__env.dt) == 0) or fin_all:
                    dst_path = dst_png_dir.joinpath("{}.png".format(out_cnt))
                    if not dst_path.exists():
                        pass
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
            learn_cnt = 0
            total_loss = []
            self.__rng, rng = jrandom.split(self.__rng)
            for i in jrandom.randint(rng, (int(self.__env.max_t / self.__env.dt),), 0, len(self.__env.experiences)):
                e = self.__env.experiences[i]
                if not e.finished:
                    s[val] = e.state[0]
                    a[val] = e.action
                    r[val] = e.reward
                    n_s[val] = e.next_state
                    n_a[val] = e.next_action
                    val += 1
                    if val >= state_shape[0]:
                        learn_cnt, loss_val = self.__env.shared_nn.update(gamma, s, a, r, n_s, n_a)
                        all_info = {}
                        all_info["episode"] = trial
                        all_info["learn_cnt"] = learn_cnt
                        all_info["total_reward_mean"] = float(total_reward_mean)
                        all_info["loss_val"] = float(loss_val)
                        all_info["J_pi"] = float(SharedNetwork.J_pi(   SharedNetwork.get_params(SharedNetwork.opt_states), s, a).mean())
                        all_info["J_q"] = float(SharedNetwork.J_q(    SharedNetwork.get_params(SharedNetwork.opt_states), s, a, r, n_s, n_a, gamma).mean())
                        all_info["log_Pi"] = float(SharedNetwork.log_Pi( SharedNetwork.get_params(SharedNetwork.opt_states), s, a).mean())
                        all_info["Q"] = float(SharedNetwork.apply_Q(SharedNetwork.get_params(SharedNetwork.opt_states), s, a).mean())
                        all_log_writer.write(all_info)
                        if verbose:
                            for value in all_info.values():
                                if isinstance(value, float):
                                    print("{:.3f}".format(value), end = ",")
                                else:
                                    print(value, end = ",")
                            print()
                        total_loss.append(loss_val)
                        val = 0
            if learn_cnt > 0:
                #print("episode={},learn_cnt={},total_reward_mean={:.3f},loss_mean={:.3f}".format(trial, learn_cnt, total_reward_mean, onp.array(total_loss).mean()))
                pass
            with open(dst_base_dir.joinpath("total.csv"), "a") as f:
                if total_log is False:
                    total_log = True
                    f.write("episode,reward,loss\n")
                f.write(str(trial))
                f.write(",")
                f.write(str(float(total_reward_mean)))
                f.write(",")
                f.write(str(onp.array(total_loss).mean()))
                f.write("\n")
            weight_path = dst_dir.joinpath("param.bin")
            self.__env.shared_nn.save(weight_path)
            #movie_cmd = "cd \"{}\" && ffmpeg -r 10 -i %d.png -vcodec libx264 -pix_fmt yuv420p \"{}\"".format(dst_png_dir.resolve(), dst_png_dir.parent.joinpath("out.mp4").resolve())
            #subprocess.run(movie_cmd)

def main():
    seed = 1
    trainer = Trainer()
    trainer.learn_episode()

if __name__ == "__main__":
    main()
