#coding: utf-8
from enum import IntEnum, auto

class EnAction(IntEnum):
    accel = 0
    omega = auto()
    num = auto()
class EnDist(IntEnum):
    mean = 0
    log_sigma = auto()
    num = auto()
class EnChannel(IntEnum):
    occupy = 0
    vy = auto()
    vx = auto()
    num = auto()
