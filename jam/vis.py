#coding: utf-8
from PIL import Image, ImageDraw, ImageOps
import numpy as onp
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
    img = Image.fromarray(onp.array(onp.zeros((pcpt_h, pcpt_w, 3), dtype = onp.uint8)))
    dr = ImageDraw.Draw(img)
    for a, agent in enumerate(agents):
        y = agent.y / map_h * pcpt_h
        x = agent.x / map_w * pcpt_w
        ry = agent.radius_m / map_h * pcpt_h 
        rx = agent.radius_m / map_w * pcpt_w
        
        py0 = onp.clip(int(y + 0.5), 0, pcpt_h - 1)
        px0 = onp.clip(int(x + 0.5), 0, pcpt_w - 1)
        py1 = onp.clip(int((y + ry * onp.sin(agent.theta)) + 0.5), 0, pcpt_h - 1)
        px1 = onp.clip(int((x + rx * onp.cos(agent.theta)) + 0.5), 0, pcpt_w - 1)
        dr.line((px0, py0, px1, py1), fill = cols[a], width = 1)
        
        py0 = onp.clip(int((y - ry) + 0.5), 0, pcpt_h - 1)
        py1 = onp.clip(int((y + ry) + 0.5), 0, pcpt_h - 1)
        px0 = onp.clip(int((x - rx) + 0.5), 0, pcpt_w - 1)
        px1 = onp.clip(int((x + rx) + 0.5), 0, pcpt_w - 1)
        dr.ellipse((px0, py0, px1, py1), outline = cols[a], width = 1)

        tgt_y = agent.tgt_y / map_h * pcpt_h 
        tgt_x = agent.tgt_x / map_w * pcpt_w
        tgt_py = onp.clip(int(tgt_y + 0.5), 0, pcpt_h)
        tgt_px = onp.clip(int(tgt_x + 0.5), 0, pcpt_w)
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

def make_all_state_img(agents, map_h, map_w, pcpt_h, pcpt_w):
    img = ImageOps.flip(make_state_img(agents, map_h, map_w, pcpt_h, pcpt_w))
    state = observe(agents, 0, map_h, map_w, pcpt_h, pcpt_w)
    occupy_img = Image.fromarray((255 * ((state[0, ::-1,:,EnChannel.occupy] + 1.0) / 2)).astype(onp.uint8))
    img = get_concat_h(img, occupy_img)
    img = get_concat_h(img, Image.fromarray((255 * ((state[0, ::-1,:,EnChannel.vy    ] + 1.5) / 3)).astype(onp.uint8)))
    img = get_concat_h(img, Image.fromarray((255 * ((state[0, ::-1,:,EnChannel.vx    ] + 1.5) / 3)).astype(onp.uint8)))
    return img

def output_state_png(agents, dst_path, map_h, map_w, pcpt_h, pcpt_w, step, dt):
    img = make_all_state_img(agents, map_h, map_w, pcpt_h, pcpt_w)
    if not dst_path.parent.exists():
        dst_path.parent.mkdir(parents = True)
    dr = ImageDraw.Draw(img)
    dr.text((0,0), "{}".format(step * dt), fill = WHITE)
    if not dst_path.parent.exists():
        dst_path.parent.mkdir(parents = True)
    img.save(dst_path)

def main():

    import time
    draw_cnt = 0
    csv_dir_path = Path(r"/home/isgsktyktt/work/tmp")
    csv_path = csv_dir_path.joinpath("learn.csv")
    print(csv_path)
    assert(csv_path.exists())

    while 1:
        plt.clf()
        fig, axs = plt.subplots(2, 2)
        df = pd.read_csv(csv_path)
        x = []
        r = []
        trials = onp.unique(df["trial"])
        for t in trials:
            episodes = onp.unique(df["episode"][df["trial"] == t])
            for e in episodes:
                tgt_idx = onp.logical_and(df["trial"] == t, df["episode"] == e)
                reward = df["total_reward_mean"][tgt_idx].mean()
                x.append(len(x))
                r.append(reward)
        markersize = 5
        axs[0, 0].plot(x, r, ".", markersize = markersize)#, label = "Reward")
        axs[0, 0].grid(True)
        #axs[0, 0].legend()
        axs[0, 0].set_title("reward")

        x = df["p_learn_cnt"]
        l = df["loss_val_pi"]
        markersize = 5
        axs[0, 1].plot(x, l, ".", markersize = markersize)#, label = "Pi-Loss")
        axs[0, 1].grid(True)
        #axs[0, 1].legend()
        axs[0, 1].set_title("pi-loss")
        #axs[2].set_ylim(-5, 0)

        for i in range(2):
            x = df["q_learn_cnt"]
            l = df["loss_val_q{}".format(i)]
            markersize = 5
            axs[1, i].plot(x, l, ".", markersize = markersize)#, label = "Q-Loss{}".format(i))
            axs[1, i].grid(True)
            #axs[1, i].legend()
            #axs[1].set_yscale("log")
            #axs[1].set_ylim(0, 5)

        #plt.show()
        if draw_cnt != l.size:
            draw_cnt = l.size
            plt.savefig(csv_dir_path.joinpath("now.png"))
        time.sleep(5)
if __name__ == "__main__":
    main()
