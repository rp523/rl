#coding: utf-8
from os import stat
import time
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from PIL import Image, ImageDraw

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
        def integral(v, a, theta, omega, t):
            return (a / (omega * omega) - (v + a * t) / omega) * jnp.sin(theta + omega * t)
        if omega != 0.0:
            dy = integral(v, a, theta, omega, dt) - integral(v, a, theta, omega, 0.0)
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
        def integral(v, a, theta, omega, t):
            return (a / (omega * omega) + (v + a * t) / omega) * jnp.cos(theta + omega * t)
        if omega != 0.0:
            dx = integral(v, a, theta, omega, dt) - integral(v, a, theta, omega, 0.0)
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
    def __make_new_pedestrian(self, old_pedestrians):
        while 1:
            rng_y, rng_x, rng_theta, self.__rng = jrandom.split(self.__rng, 4)
            tgt_y, y = jrandom.uniform(rng_y, (2,), minval = Pedestrian.radius_m, maxval = self.__map_h - Pedestrian.radius_m)
            tgt_x, x = jrandom.uniform(rng_x, (2,), minval = Pedestrian.radius_m, maxval = self.__map_w - Pedestrian.radius_m)
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
        n_ped = jrandom.randint(_rng, (1,), 1, self.__n_ped_max + 1)

        objects = []
        for _ in range(int(n_ped)):
            new_ped = self.__make_new_pedestrian(objects)
            objects.append(new_ped)
        assert(len(objects) == n_ped)
        return objects
    def evolve(self, objects, actions):
        for obj, act in zip(objects, actions):
            a, omega = act
            obj.y = obj.calc_new_y(a, omega)
            obj.x = obj.calc_new_x(a, omega)
            obj.v = obj.calc_new_v(a)
            obj.theta = obj.calc_new_theta(omega)
        return objects

def observe(pedestrians, map_h, map_w, pcpt_h, pcpt_w):
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    YELLOW = (255, 255, 0)
    cols = [RED, BLUE, GREEN, YELLOW]
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
    return occupy

if __name__ == "__main__":
    rng = jrandom.PRNGKey(0)
    while 1:
        _rng, rng = jrandom.split(rng, 2)
        env = Environment(_rng, 100.0, 100.0, 4)
        peds = env.make_init_state()
        #peds = env.evolve(peds, [(0.0, 0.0)] * len(peds))
        observe(peds, 100.0, 100.0, 256, 256).show()
        print(input())
        time.sleep(0.1)