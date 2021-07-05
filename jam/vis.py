#coding: utf-8
from PIL import Image, ImageDraw, ImageOps
import jax.numpy as jnp
from common import *
from observer import observe
from matplotlib import pyplot as plt
import pandas as pd
from pathlib import Path

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
def make_state_img(agents, map_h, map_w, pcpt_h, pcpt_w):
    cols = (WHITE, RED, YELLOW, GREEN, BLUE, SKY, PINK, ORANGE, PURPLE, BROWN)
    img = Image.fromarray(onp.array(jnp.zeros((pcpt_h, pcpt_w, 3), dtype = jnp.uint8)))
    dr = ImageDraw.Draw(img)
    for a, agent in enumerate(agents):
        y = agent.y / map_h * pcpt_h
        x = agent.x / map_w * pcpt_w
        ry = agent.radius_m / map_h * pcpt_h 
        rx = agent.radius_m / map_w * pcpt_w
        
        py0 = jnp.clip(int(y + 0.5), 0, pcpt_h - 1)
        px0 = jnp.clip(int(x + 0.5), 0, pcpt_w - 1)
        py1 = jnp.clip(int((y + ry * onp.sin(agent.theta)) + 0.5), 0, pcpt_h - 1)
        px1 = jnp.clip(int((x + rx * onp.cos(agent.theta)) + 0.5), 0, pcpt_w - 1)
        dr.line((px0, py0, px1, py1), fill = cols[a], width = 1)
        
        py0 = jnp.clip(int((y - ry) + 0.5), 0, pcpt_h - 1)
        py1 = jnp.clip(int((y + ry) + 0.5), 0, pcpt_h - 1)
        px0 = jnp.clip(int((x - rx) + 0.5), 0, pcpt_w - 1)
        px1 = jnp.clip(int((x + rx) + 0.5), 0, pcpt_w - 1)
        dr.ellipse((px0, py0, px1, py1), outline = cols[a], width = 1)

        tgt_y = agent.tgt_y / map_h * pcpt_h 
        tgt_x = agent.tgt_x / map_w * pcpt_w
        tgt_py = jnp.clip(int(tgt_y + 0.5), 0, pcpt_h)
        tgt_px = jnp.clip(int(tgt_x + 0.5), 0, pcpt_w)
        lin_siz = 5
        dr.line((tgt_px - lin_siz, tgt_py - lin_siz, tgt_px + lin_siz, tgt_py + lin_siz), width = 1, fill = cols[a])
        dr.line((tgt_px - lin_siz, tgt_py + lin_siz, tgt_px + lin_siz, tgt_py - lin_siz), width = 1, fill = cols[a])
    dr.rectangle((0, 0, pcpt_w - 1, pcpt_h - 1), outline = WHITE)
    return img
def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst
def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

def output_state_png(agents, dst_path, map_h, map_w, step, dt):
    pcpt_h = 128
    pcpt_w = 128
    img = ImageOps.flip(make_state_img(agents, map_h, map_w, pcpt_h, pcpt_w))
    img = get_concat_h(img, Image.fromarray((255 * ((observe(agents, 0, map_h, map_w, pcpt_h, pcpt_w)[::-1,:,EnChannel.occupy] + 1.0) / 2)).astype(jnp.uint8)))
    img = get_concat_h(img, Image.fromarray((255 * ((observe(agents, 0, map_h, map_w, pcpt_h, pcpt_w)[::-1,:,EnChannel.vy    ] + 1.5) / 3)).astype(jnp.uint8)))
    img = get_concat_h(img, Image.fromarray((255 * ((observe(agents, 0, map_h, map_w, pcpt_h, pcpt_w)[::-1,:,EnChannel.vx    ] + 1.5) / 3)).astype(jnp.uint8)))

    if not dst_path.parent.exists():
        dst_path.parent.mkdir(parents = True)
    dr = ImageDraw.Draw(img)
    dr.text((0,0), "{}".format(step * dt), fill = WHITE)
    if not dst_path.parent.exists():
        dst_path.parent.mkdir(parents = True)
    img.save(dst_path)

if __name__ == "__main__":
    csv_path = Path(r"/home/isgsktyktt/work/tmp/loss.csv")
    df = pd.read_csv(csv_path)
    plt.clf()
    x = []

    if 0:
        y0 = []
        y1 = []
        for e in jnp.unique(df["episode"]):
            x.append(e)
            y0.append((df["loss_val"][df["episode"] == e]).mean())
            y1.append((df["total_reward_mean"][df["episode"] == e]).mean())
    else:
        x = df["learn_cnt"]
        y0 = df["loss_val"]
        y1 = df["total_reward_mean"]
    markersize = 5
    plt.plot(x, y0, ".", markersize = markersize, label = "loss")
    plt.plot(x, y1, ".", markersize = markersize, label = "reward")
    plt.grid(True)
    plt.legend()
    plt.show()