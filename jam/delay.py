#coding: utf-8
class DelayGen:
    def __init__(self, decay_rate, accum_max):
        self.__reward = accum_max * (1.0 - decay_rate)
    def __call__(self):
        return self.__reward
