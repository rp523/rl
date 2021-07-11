#coding: utf-8
from PIL import Image, ImageDraw
import numpy as onp
from common import *

def observe(agents, agent_idx, map_h, map_w, pcpt_h, pcpt_w):
    state = onp.zeros((1, pcpt_h, pcpt_w, EnChannel.num), dtype = onp.float32)

    own_obs_y = 0.5 * map_h
    own_obs_x = 0.5 * map_w

    own = agents[agent_idx]

    wall_img = Image.fromarray(state[0,:,:,0].astype(onp.uint8))
    dr = ImageDraw.Draw(wall_img)
    for wall_y0, wall_x0, wall_y1, wall_x1 in [
                    [  0.0,   0.0,   0.0, map_w],
                    [map_h,   0.0, map_h, map_w],
                    [  0.0,   0.0, map_h,   0.0],
                    [  0.0, map_w, map_h, map_w],
                    ]:
        rel_y0 = wall_y0 - own.y
        rel_y1 = wall_y1 - own.y
        rel_x0 = wall_x0 - own.x
        rel_x1 = wall_x1 - own.x
        obs_y0, obs_x0 = rotate(rel_y0, rel_x0, 0.5 * onp.pi - own.theta)
        obs_y1, obs_x1 = rotate(rel_y1, rel_x1, 0.5 * onp.pi - own.theta)
        obs_py0 = int((obs_y0 + own_obs_y) / map_h * pcpt_h + 0.5)
        obs_py1 = int((obs_y1 + own_obs_y) / map_h * pcpt_h + 0.5)
        obs_px0 = int((obs_x0 + own_obs_x) / map_w * pcpt_w + 0.5)
        obs_px1 = int((obs_x1 + own_obs_x) / map_w * pcpt_w + 0.5)
        dr.line((obs_px0, obs_py0, obs_px1, obs_py1), fill = (1,), width = 1)
    state[0, :, :, EnChannel.occupy] = onp.asarray(wall_img).astype(onp.float32)
    
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
            state[0, max(obs_py0, 0) : min(obs_py1 + 1, pcpt_h), max(obs_px0, 0) : min(obs_px1 + 1, pcpt_w), EnChannel.occupy] = 1.0
            state[0, max(obs_py0, 0) : min(obs_py1 + 1, pcpt_h), max(obs_px0, 0) : min(obs_px1 + 1, pcpt_w), EnChannel.vy    ] = other.v * onp.cos(own.theta - other.theta)
            state[0, max(obs_py0, 0) : min(obs_py1 + 1, pcpt_h), max(obs_px0, 0) : min(obs_px1 + 1, pcpt_w), EnChannel.vx    ] = other.v * onp.sin(own.theta - other.theta)

    rel_y = own.tgt_y - own.y
    rel_x = own.tgt_x - own.x
    obs_y, obs_x = rotate(rel_y, rel_x, 0.5 * onp.pi - own.theta)
    obs_py0 = int((obs_y - own.radius_m + own_obs_y) / map_h * pcpt_h + 0.5)
    obs_py1 = int((obs_y + own.radius_m + own_obs_y) / map_h * pcpt_h + 0.5)
    obs_px0 = int((obs_x - own.radius_m + own_obs_x) / map_w * pcpt_w + 0.5)
    obs_px1 = int((obs_x + own.radius_m + own_obs_x) / map_w * pcpt_w + 0.5)
    if  (obs_py0 <= pcpt_h - 1) and (0 <= obs_py1) and \
        (obs_px0 <= pcpt_w - 1) and (0 <= obs_px1):
        state[0, max(obs_py0, 0) : min(obs_py1 + 1, pcpt_h), max(obs_px0, 0) : min(obs_px1 + 1, pcpt_w), EnChannel.occupy] = - 1.0

    obs_y = 0.0
    obs_x = 0.0
    obs_py0 = int((obs_y - own.radius_m + own_obs_y) / map_h * pcpt_h + 0.5)
    obs_py1 = int((obs_y + own.radius_m + own_obs_y) / map_h * pcpt_h + 0.5)
    obs_px0 = int((obs_x - own.radius_m + own_obs_x) / map_w * pcpt_w + 0.5)
    obs_px1 = int((obs_x + own.radius_m + own_obs_x) / map_w * pcpt_w + 0.5)
    if  (obs_py0 <= pcpt_h - 1) and (0 <= obs_py1) and \
        (obs_px0 <= pcpt_w - 1) and (0 <= obs_px1):
        state[0, max(obs_py0, 0) : min(obs_py1 + 1, pcpt_h), max(obs_px0, 0) : min(obs_px1 + 1, pcpt_w), EnChannel.vy    ] = own.v
    
    return state

def rotate(y, x, theta):
    rot_x = onp.cos(theta) * x - onp.sin(theta) * y
    rot_y = onp.sin(theta) * x + onp.cos(theta) * y
    return rot_y, rot_x

if __name__ == "__main__":
    img = Image.fromarray(onp.ones((100,100), dtype = onp.uint8) * 122)
    dr = ImageDraw.Draw(img)
    obs_py0 = -30
    obs_py1 = 30
    obs_px0 = -30
    obs_px1 = 30
    dr.ellipse((obs_px0, obs_py0, obs_px1, obs_py1), fill = (255), width = 1)
    img.show()
