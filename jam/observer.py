#coding: utf-8
import numpy as onp
from common import *

def observe(agents, agent_idx, map_h, map_w, pcpt_h, pcpt_w):
    state = onp.zeros((pcpt_h, pcpt_w, EnChannel.num), dtype = onp.float32)

    own_obs_y = 0.5 * map_h
    own_obs_x = 0.5 * map_w
    
    own = agents[agent_idx]
    for other in (agents[:agent_idx] + agents[agent_idx + 1:]):
        rel_y = other.y - own.y
        rel_x = other.x - own.x
        obs_y, obs_x = rotate(rel_y, rel_x, 0.5 * onp.pi - own.theta)
        obs_py0 = int((obs_y - other.radius_m + own_obs_y) / map_h * pcpt_h + 0.5)
        obs_py1 = int((obs_y + other.radius_m + own_obs_y) / map_h * pcpt_h + 0.5)
        obs_px0 = int((obs_x - other.radius_m + own_obs_x) / map_w * pcpt_w + 0.5)
        obs_px1 = int((obs_x + other.radius_m + own_obs_x) / map_w * pcpt_w + 0.5)
        if  (obs_py0 <= pcpt_h - 1) and (0 <= obs_py1) and \
            (obs_px0 <= pcpt_w - 1) and (0 <= obs_px1):
            state[max(obs_py0, 0) : min(obs_py1 + 1, pcpt_h), max(obs_px0, 0) : min(obs_px1 + 1, pcpt_w), EnChannel.occupy] = 1.0
            state[max(obs_py0, 0) : min(obs_py1 + 1, pcpt_h), max(obs_px0, 0) : min(obs_px1 + 1, pcpt_w), EnChannel.vy    ] = other.v * onp.cos(own.theta - other.theta)
            state[max(obs_py0, 0) : min(obs_py1 + 1, pcpt_h), max(obs_px0, 0) : min(obs_px1 + 1, pcpt_w), EnChannel.vx    ] = other.v * onp.sin(own.theta - other.theta)

    rel_y = own.tgt_y - own.y
    rel_x = own.tgt_x - own.x
    obs_y, obs_x = rotate(rel_y, rel_x, 0.5 * onp.pi - own.theta)
    obs_py0 = int((obs_y - own.radius_m + own_obs_y) / map_h * pcpt_h + 0.5)
    obs_py1 = int((obs_y + own.radius_m + own_obs_y) / map_h * pcpt_h + 0.5)
    obs_px0 = int((obs_x - own.radius_m + own_obs_x) / map_w * pcpt_w + 0.5)
    obs_px1 = int((obs_x + own.radius_m + own_obs_x) / map_w * pcpt_w + 0.5)
    if  (obs_py0 <= pcpt_h - 1) and (0 <= obs_py1) and \
        (obs_px0 <= pcpt_w - 1) and (0 <= obs_px1):
        state[max(obs_py0, 0) : min(obs_py1 + 1, pcpt_h), max(obs_px0, 0) : min(obs_px1 + 1, pcpt_w), EnChannel.occupy] = - 1.0

    obs_y = 0.0
    obs_x = 0.0
    obs_py0 = int((obs_y - own.radius_m + own_obs_y) / map_h * pcpt_h + 0.5)
    obs_py1 = int((obs_y + own.radius_m + own_obs_y) / map_h * pcpt_h + 0.5)
    obs_px0 = int((obs_x - own.radius_m + own_obs_x) / map_w * pcpt_w + 0.5)
    obs_px1 = int((obs_x + own.radius_m + own_obs_x) / map_w * pcpt_w + 0.5)
    if  (obs_py0 <= pcpt_h - 1) and (0 <= obs_py1) and \
        (obs_px0 <= pcpt_w - 1) and (0 <= obs_px1):
        state[max(obs_py0, 0) : min(obs_py1 + 1, pcpt_h), max(obs_px0, 0) : min(obs_px1 + 1, pcpt_w), EnChannel.vy    ] = own.v
    
    return state

def rotate(y, x, theta):
    rot_x = onp.cos(theta) * x - onp.sin(theta) * y
    rot_y = onp.sin(theta) * x + onp.cos(theta) * y
    return rot_y, rot_x