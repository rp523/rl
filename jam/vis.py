#coding: utf-8
from PIL import Image, ImageDraw, ImageOps
import numpy as onp
from common import *
from observer import observe
from matplotlib import pyplot as plt
import pandas as pd
from pathlib import Path
import time

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

def main(org_draw_cnt):

    csv_dir_path = None
    for csv_dir_path_koho in Path(r"/home/isgsktyktt/work/outputs").rglob("*"):
        if csv_dir_path_koho.is_dir():
            if csv_dir_path_koho.joinpath("out").exists():
                if csv_dir_path is None:
                    csv_dir_path = csv_dir_path_koho
                else:
                    if (csv_dir_path_koho.stat().st_ctime > csv_dir_path.stat().st_ctime):
                        csv_dir_path = csv_dir_path_koho
    
    csv_path_list = []
    for csv_path in csv_dir_path.rglob("*/learn.csv"):
        if csv_path.is_file():
            csv_path_list.append(csv_path)

    fig, axs = None, None
    for csv_path in csv_path_list:

        row = 0
        if csv_path.parents[3].name == "outputs":
            idx = 0
        elif csv_path.parents[3].name == "multirun":
            idx = int(csv_path.parents[1].name)
        else:
            assert(0)

        df = pd.read_csv(csv_path)
        if (df[df.columns[0]].size == org_draw_cnt):
            return org_draw_cnt
        
        trials = onp.unique(df["trial"])
        uni = 16*16
        markersize = 2

        if 1:
            plt.clf()
            plt.figure(figsize=(12.0, 8.0))
            _y = []
            for t in trials:
                episodes = onp.unique(df["episode"][df["trial"] == t])
                for e in episodes:
                    play_csv_path = csv_path.parent.joinpath("play", "{}_{}.csv".format(t, e))
                    play_df = pd.read_csv(play_csv_path)
                    _y.append(play_df["reward0"].sum())
            _y = onp.array(_y)
            y = []
            for i in range(len(_y)):
                if i > uni:
                    y.append(_y[i-uni:i].mean())
            def fx(y, _y):
                return (onp.arange(len(y)) - len(y) + len(_y))
            plt.plot(fx(_y,_y), _y, ".", markersize = markersize)#, label = "Reward")
            plt.plot(fx(y,_y), y, ".", markersize = markersize)#, label = "Reward")
            plt.grid(True)
            #plt.xscale("log")
            plt.xlim(500)
            plt.ylim(-40, -0)

        else:
            if (fig is None) and (axs is None):
                plt.clf()
                fig, axs = plt.subplots(7, max(2, len(csv_path_list)),
                    figsize=(12.0, 8.0)
                )

            x = []
            y = []
            for t in trials:
                episodes = onp.unique(df["episode"][df["trial"] == t])
                for e in episodes:
                    play_csv_path = csv_path.parent.joinpath("play", "{}_{}.csv".format(t, e))
                    play_df = pd.read_csv(play_csv_path)
                    x.append(len(x))
                    y.append(play_df["reward0"].sum())
            axs[row, idx].plot(x, y, ".", markersize = markersize)#, label = "Reward")
            axs[row, idx].grid(True)
            #axs[row, idx].set_ylim(-30, 1)
            row += 1

            _y = []
            for t in trials:
                episodes = onp.unique(df["episode"][df["trial"] == t])
                for e in episodes:
                    play_csv_path = csv_path.parent.joinpath("play", "{}_{}.csv".format(t, e))
                    play_df = pd.read_csv(play_csv_path)
                    _y.append(play_df["reward0"].sum())
            _y = onp.array(_y)
            x = []
            y = []
            for i in range(len(_y)):
                if i > uni:
                    x.append(i)
                    y.append(_y[i-uni:i].mean())
            axs[row, idx].plot(x, y, ".", markersize = markersize)#, label = "Reward")
            axs[row, idx].grid(True)
            #axs[row, idx].set_ylim(-2, 1)
            row += 1

            for i in range(2):
                x = []
                _y1 = []
                _y2 = []
                y1 = []
                y2 = []
                for t in trials:
                    episodes = onp.unique(df["episode"][df["trial"] == t])
                    for e in episodes:
                        play_csv_path = csv_path.parent.joinpath("play", "{}_{}.csv".format(t, e))
                        play_df = pd.read_csv(play_csv_path)
                        x.append(len(x))
                        _y1.append(play_df["obj0_sigma{}".format(i)].mean())
                        _y2.append(play_df["obj0_mean{}".format(i)].mean())
                        y1.append(onp.array(_y1[max(0,-1-uni):]).mean())
                        y2.append(onp.array(_y2[max(0,-1-uni):]).mean())
                axs[row, idx].plot(x, y1, ".", markersize = markersize)#, label = "Reward")
                axs[row, idx].plot(x, y2, ".", markersize = markersize)#, label = "Reward")
                axs[row, idx].grid(True)
                #axs[row, idx].set_ylim(-2, 1)
                row += 1

            x = df["q_learn_cnt"]
            l = df["temperature"]
            axs[row, idx].plot(x, l, ".", markersize = markersize)#, label = "Pi-Loss")
            axs[row, idx].grid(True)
            axs[row, idx].set_yscale("log")
            row += 1
            
            '''
            for action in ["accel", "omega"]:
                x = []
                y = []
                for t in trials:
                    episodes = onp.unique(df["episode"][df["trial"] == t])
                    for e in episodes:
                        play_csv_path = csv_path.parent.joinpath("play", "{}_{}.csv".format(t, e))
                        play_df = pd.read_csv(play_csv_path)
                        y.append(play_df["{}_sigma0".format(action)].mean())
                        x.append(t)
                axs[row, idx].plot(x, y, ".", markersize = markersize)
                axs[row, idx].grid(True)
                #axs[row, idx].set_yscale("log")
                #axs[row, idx].set_ylim(0.75, 1.0)
            row += 1
            '''

            x = []
            _y = df["loss_val_pi"]
            y = []
            for j in range(len(_y)):
                x.append(j)
                y.append(onp.array(_y[max(0,j-uni):j]).mean())
            pi_min_idx = 0
            axs[row, idx].plot(x[pi_min_idx:], y[pi_min_idx:], ".", markersize = markersize)#, label = "Pi-Loss")
            axs[row, idx].grid(True)
            #axs[row, idx].set_ylim(0)
            #axs[row, idx].set_title("pi-loss")
            row += 1

            for i in range(2):
                x = []
                y = []
                _y = df["loss_val_q{}".format(i)]
                for j in range(len(_y)):
                    x.append(j)
                    y.append(onp.array(_y[max(0,j-uni):j]).mean())
                axs[row, idx].plot(x, y, ".", markersize = markersize)#, label = "Q-Loss{}".format(i))
                axs[row, idx].grid(True)
                axs[row, idx].set_yscale("log")
            row += 1
        
    #plt.show();exit()
    plt.savefig(csv_dir_path.joinpath("now.png"))
    return df[df.columns[0]].size

if __name__ == "__main__":
    draw_cnt = 0
    while 1:
        draw_cnt = main(draw_cnt)
        time.sleep(5)
