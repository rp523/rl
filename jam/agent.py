#coding: utf-8
import jax.numpy as jnp
class ObjectBase:
    __eps = 1E-2
    def __init__(self, y, x, theta, dt, v_max, a_max):
        self.__y = y
        self.__x = x
        self.__init_y = y
        self.__init_x = x
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
    def init_y(self):
        return self.__init_y
    @property
    def init_x(self):
        return self.__init_x
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
            if dt_m > ObjectBase.__eps:
                dy = ObjectBase.__calc_dy(v, a, theta, omega, dt_m, v_max, a_max)
            if dt - dt_m > ObjectBase.__eps:
                dy += ObjectBase.__calc_dy(v_max, 0.0, theta + omega * dt_m, omega, dt - dt_m, v_max, a_max)
        elif v + a * dt < 0.0:  # min speed limit
            dt_m = v / a
            if dt_m > 0.0:
                dy = ObjectBase.__calc_dy(v, a, theta, omega, dt_m, v_max, a_max)
        else:
            dy = ObjectBase.__calc_dy_impl(v, a, theta, omega, dt)
        return dy
    @staticmethod
    def __calc_dy_impl(v, a, theta, omega, dt):
        def integral_y(_v, _a, _theta, _omega, _dt):
            return _a / (_omega * _omega) * jnp.sin(_theta + _omega * _dt) - (_v + _a * _dt) / _omega * jnp.cos(_theta + _omega * _dt)
        if abs(omega) > ObjectBase.__eps:
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
            if dt_m > ObjectBase.__eps:
                dx = ObjectBase.__calc_dx(v, a, theta, omega, dt_m, v_max, a_max)
            if dt - dt_m > ObjectBase.__eps:
                dx += ObjectBase.__calc_dx(v_max, 0.0, theta + omega * dt_m, omega, dt - dt_m, v_max, a_max)
        elif v + a * dt < 0.0:  # min speed limit
            dt_m = v / a
            if dt_m > 0.0:
                dx = ObjectBase.__calc_dx(v, a, theta, omega, dt_m, v_max, a_max)
        else:
            dx = ObjectBase.__calc_dx_impl(v, a, theta, omega, dt)
        return dx
    @staticmethod
    def __calc_dx_impl(v, a, theta, omega, dt):
        def integral_x(_v, _a, _theta, _omega, _dt):
            return _a / (_omega * _omega) * jnp.cos(_theta + _omega * _dt) + (_v + _a * _dt) / _omega * jnp.sin(_theta + _omega * _dt)
        if abs(omega) > ObjectBase.__eps:
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
            self.stop()
    def stop(self):
            self.v = 0.0

class Pedestrian(ObjectBase):
    def __init__(self, y, x, theta, dt):
        v_max = 1.4
        a_max = v_max / 1.0
        super().__init__(y, x, theta, dt, v_max, a_max)

class PedestrianAgent(Pedestrian):
    def __init__(self, tgt_y, tgt_x, y, x, theta, dt):
        super().__init__(y, x, theta, dt)
        self.__tgt_y = tgt_y
        self.__tgt_x = tgt_x
        self.__radius_m = 0.5
        self.reserved_action = None
    # getter
    @property
    def tgt_y(self):
        return self.__tgt_y
    @property
    def tgt_x(self):
        return self.__tgt_x
    @property
    def radius_m(self):
        return self.__radius_m
    
    def reached_goal(self):
        reached = False
        if  (abs(self.y - self.tgt_y) < self.radius_m) and \
            (abs(self.x - self.tgt_x) < self.radius_m):
            reached = True
        return reached
    
    def hit_with(self, other):
        hit = False
        if  (abs(self.y - other.tgt_y) < self.radius_m) and \
            (abs(self.x - other.tgt_x) < self.radius_m):
            hit = True
        return hit

