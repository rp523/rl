#coding: utf-8
from os import stat
import time
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path

class ObjectBase:
    def __init__(self, tgt_y, tgt_x, y, x, theta, dt, v_max, a_max):
        self.__tgt_y = tgt_y
        self.__tgt_x = tgt_x
        self.__y = y
        self.__x = x
        self.__theta = theta
        self.__v = 0.0

        self.__dt = dt
        self.__v_max = v_max
        self.__a_max = a_max

    # getter
    @property
    def tgt_y(self):
        return self.__tgt_y
    @property
    def tgt_x(self):
        return self.__tgt_x
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
        return self.__y + ObjectBase.__calc_dy(self.v, a, self.theta, omega, self.__dt, self.__v_max, self.__a_max)
    @staticmethod
    def __calc_dy(v, a, theta, omega, dt, v_max, a_max):
        dy = 0.0
        a = np.clip(a, -a_max, a_max)
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
        if omega != 0.0:
            dy = integral_y(v, a, theta, omega, dt) - integral_y(v, a, theta, omega, 0.0)
        else:
            dy = (v + 0.5 * a * dt) * dt * jnp.sin(theta)
        return dy

    # x-position
    def calc_new_x(self, a, omega):
        return self.__x + ObjectBase.__calc_dx(self.v, a, self.theta, omega, self.__dt, self.__v_max, self.__a_max)
    @staticmethod
    def __calc_dx(v, a, theta, omega, dt, v_max, a_max):
        dx = 0.0
        a = np.clip(a, -a_max, a_max)
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
        if omega != 0.0:
            dx = integral_x(v, a, theta, omega, dt) - integral_x(v, a, theta, omega, 0.0)
        else:
            dx = (v + 0.5 * a * dt) * dt * jnp.cos(theta)
        return dx

    # speed
    def calc_new_v(self, a):
        return ObjectBase.__calc_new_v(self.v, a, self.__dt, self.__v_max, self.__a_max)
    @staticmethod
    def __calc_new_v(v, a, dt, v_max, a_max):
        a = np.clip(a, -a_max, a_max)
        return jnp.clip(v + a * dt, 0.0, v_max)

    # rotation
    def calc_new_theta(self, omega):
        return ObjectBase.__calc_new_theta(self.theta, omega, self.__dt)
    @staticmethod
    def __calc_new_theta(theta, omega, dt):
        new_theta = theta + omega * dt
        if new_theta < 0.0:
            new_theta += 2.0 * np.pi
        elif new_theta > 2.0 * np.pi:
            new_theta -= 2.0 * np.pi
        return new_theta
    # update motion status
    def evolve(self, a, omega, y_min, y_max, x_min, x_max):
        new_theta = self.calc_new_theta(omega)
        new_y = self.calc_new_y(a, omega)
        new_x = self.calc_new_x(a, omega)
        new_v = self.calc_new_v(a)

        self.theta = new_theta
        if (new_y > y_min) and (new_y < y_max) and (new_x > x_min) and (new_x < x_max):
            self.y = new_y
            self.x = new_x
            self.v = new_v
        else:
            self.v = 0.0

class Pedestrian(ObjectBase):
    radius_m = 0.5
    def __init__(self, tgt_y, tgt_x, y, x, theta, dt):
        v_max = 1.4
        a_max = v_max / 1.0
        super().__init__(tgt_y, tgt_x, y, x, theta, dt, v_max, a_max)

class Environment:
    def __init__(self, rng, map_h, map_w, n_ped_max):
        self.__rng = rng
        self.__n_ped_max = n_ped_max
        self.__map_h = map_h
        self.__map_w = map_w
        self.__dt = 0.05
    @property
    def n_ped_max(self):
        return self.__n_ped_max
    @property
    def map_h(self):
        return self.__map_h
    @property
    def map_w(self):
        return self.__map_w

    def __make_new_pedestrian(self, old_pedestrians):
        while 1:
            rng_y, rng_x, rng_theta, self.__rng = jrandom.split(self.__rng, 4)
            tgt_y, y = jrandom.uniform(rng_y, (2,), minval = Pedestrian.radius_m, maxval = self.map_h - Pedestrian.radius_m)
            tgt_x, x = jrandom.uniform(rng_x, (2,), minval = Pedestrian.radius_m, maxval = self.map_w - Pedestrian.radius_m)
            theta = jrandom.uniform(rng_theta, (1,), minval = 0.0, maxval = 2.0 * jnp.pi)[0]
            new_ped = Pedestrian(tgt_y, tgt_x, y, x, theta, self.__dt)

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

    def make_init_state(self):
        _rng, self.__rng = jrandom.split(self.__rng, 2)
        n_ped = jrandom.randint(_rng, (1,), 1, self.n_ped_max + 1)

        objects = []
        for _ in range(int(n_ped)):
            new_ped = self.__make_new_pedestrian(objects)
            objects.append(new_ped)
        assert(len(objects) == n_ped)
        return objects
    def evolve(self, objects, actions):
        for obj, act in zip(objects, actions):
            a, omega = act
            y_min = obj.radius_m
            y_max = self.map_h - obj.radius_m
            x_min = obj.radius_m
            x_max = self.map_w - obj.radius_m
            obj.evolve(a, omega, y_min, y_max, x_min, x_max)
            
        return objects

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
def observe(pedestrians, map_h, map_w, pcpt_h, pcpt_w):
    cols = (WHITE, RED, YELLOW, GREEN, BLUE, SKY, PINK, ORANGE, PURPLE, BROWN)
    occupy = Image.fromarray(np.zeros((pcpt_h, pcpt_w, 3), dtype = np.uint8))
    dr = ImageDraw.Draw(occupy)
    for p, ped in enumerate(pedestrians):
        y = ped.y / map_h * pcpt_h
        x = ped.x / map_w * pcpt_w
        ry = ped.radius_m / map_h * pcpt_h 
        rx = ped.radius_m / map_w * pcpt_w
        
        py0 = np.clip(int((y - ry) + 0.5), 0, pcpt_h)
        py1 = np.clip(int((y + ry) + 0.5) + 1, 0, pcpt_h)
        px0 = np.clip(int((x - rx) + 0.5), 0, pcpt_w)
        px1 = np.clip(int((x + rx) + 0.5) + 1, 0, pcpt_w)
        dr.rectangle((px0, py0, px1, py1), fill = cols[p])

        ty = ped.tgt_y / map_h * pcpt_h 
        tx = ped.tgt_x / map_w * pcpt_w
        pty = np.clip(int(ty + 0.5), 0, pcpt_h)
        ptx = np.clip(int(tx + 0.5), 0, pcpt_w)
        lin_siz = 2
        dr.line((ptx - lin_siz, pty - lin_siz, ptx + lin_siz, pty + lin_siz), width = 1, fill = cols[p])
        dr.line((ptx - lin_siz, pty + lin_siz, ptx + lin_siz, pty - lin_siz), width = 1, fill = cols[p])
    dr.rectangle((0, 0, pcpt_w - 1, pcpt_h - 1), outline = WHITE)
    return occupy

def test():
    rng = jrandom.PRNGKey(0)
    _rng, rng = jrandom.split(rng, 2)
    map_h = 50.0
    map_w = 50.0
    env = Environment(_rng, map_h, map_w, 4)
    peds = env.make_init_state()
    dst_dir = Path("tmp")
    f = None
    f = open(dst_dir.joinpath("log.csv"), "w")
    if f is not None:
        f.write("frame,")
        for ped in peds:
            f.write("y,x,v,theta,")
        f.write("\n")

    for i in range(20 * 100):
        peds = env.evolve(peds, [(1.0, 0.1)] * len(peds))

        dst_path = dst_dir.joinpath("png", "{}.png".format(i))
        print(dst_path.resolve())
        if not dst_path.exists():
            img = observe(peds, map_h, map_w, 256, 256)
            if not dst_path.parent.exists():
                dst_path.parent.mkdir(parents = True)
            dr = ImageDraw.Draw(img)
            dr.text((0,0), "{}".format(i), fill = WHITE)
            img.save(dst_path)
            if f is not None:
                f.write("{},".format(i))
                for ped in peds:
                    f.write("{},{},{},{},".format(ped.y, ped.x, ped.v, ped.theta))
                f.write("\n")


#coding: utf-8
import subprocess
import sys
sys.path.append("jax")
import time
from pathlib import Path
from PIL import Image
import numpy as np
from enum import IntEnum, auto, unique
import jax
import jax.numpy as jnp
import hydra
from jax.experimental.stax import serial, parallel, Dense, Tanh, Conv, Flatten, FanOut, FanInSum, Identity, BatchNorm
from jax.experimental.optimizers import adam
from dataset.fashion_mnist import FashionMnist

def nn(cn):
    return serial(  Conv( 8, (7, 7), (1, 1), "VALID"), Tanh,# 22
                    Conv(16, (5, 5), (1, 1), "VALID"), Tanh,# 18
                    Conv(16, (3, 3), (1, 1), "VALID"), Tanh,# 16
                    Conv(16, (3, 3), (1, 1), "VALID"), Tanh,# 14
                    Conv(16, (3, 3), (1, 1), "VALID"), Tanh,# 12
                    Conv(16, (3, 3), (1, 1), "VALID"), Tanh,# 10
                    Conv(16, (3, 3), (1, 1), "VALID"), Tanh,#  8
                    Conv(32, (3, 3), (1, 1), "VALID"), Tanh,#  6
                    Conv(32, (3, 3), (1, 1), "VALID"), Tanh,#  4
                    Conv(cn, (4, 4), (1, 1), "VALID"), Tanh,#  1
                    Flatten,
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