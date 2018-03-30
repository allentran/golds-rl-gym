import argparse
import json

import gym
import numpy as np
import matplotlib
import matplotlib.cm as cm
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import fed_gym.envs.multiagent
from IPython import display
from matplotlib.animation import FuncAnimation



def hist_calc(x,Nsize):
    N=x.shape[0]
    u=np.zeros((Nsize,Nsize))
    amin=np.amin(x, axis=0)
    amax=np.amax(x, axis=0)
    xmin0=amin[0]
    xmax0=amax[0]+.000001
    xmin1=amin[1]
    xmax1=amax[1]+.000001
    for j in range(N):
        xs=np.int(np.floor(Nsize*(x[j][0]-xmin0)/(xmax0-xmin0)))
        ys=np.int(np.floor(Nsize*(x[j][1]-xmin1)/(xmax1-xmin1)))

        u[xs,ys]+=1
    return u

def update(idx):
    global x, xa

    action = actions[idx]
    next_state, reward, done, _ = env.step(action)
    x = next_state[0]
    xa = next_state[1]

    make_plot(x, xa) # plots a movie, turn off to speed up
    return reward


def make_plot(x, xa):
    plt.clf()
    plt.scatter(x[:, 0], x[:, 1], c=np.arange(len(x)), cmap='Reds')
    plt.scatter(xa[:, 0], xa[:, 1], c='dodgerblue')
    plt.ylim((0, 4))
    plt.xlim((0, 9))
    plt.yticks([])
    plt.xticks([])
    display.clear_output(wait=True)
    display.display(plt.gcf())


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--actions', type=str, default='swarm-eval.json')
    return parser


if __name__ == '__main__':
    args = get_arg_parser().parse_args()
    with open(args.actions, 'r') as f:
        actions_json = json.load(f)

    actions = np.array(actions_json['actions'])
    env = gym.envs.make("Swarm-eval-v0")
    states = env.reset()
    x = states[0]
    xa = states[1]
    plt.style.use('ggplot')
    fig, ax = plt.subplots()

    # t = 0
    # rewards = []
    # for t in range(len(actions)):
    #     rewards.append(update(t))
    # print(np.mean(rewards))

    anim = FuncAnimation(fig, update, frames=np.arange(0, 128), interval=50)
    anim.save('line-random.gif', dpi=80, writer='imagemagick')


