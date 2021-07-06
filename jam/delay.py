#coding: utf-8
class DelayGen:
    def __init__(self, decay_rate, accum_max):
        self.__init_val = accum_max * (1.0 - decay_rate)
        self.__decay_rate = decay_rate
        self.__reward = self.__init_val
    def __call__(self):
        return self.__reward
    def step(self):
        self.__reward = self.__reward * self.__decay_rate
    def reset(self):
        self.__reward = self.__init_val