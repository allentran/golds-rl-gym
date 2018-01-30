import json

import numpy as np
import gym

from fed_gym.envs.fed_env import register_solow_env


def make_env(p, q):
    return gym.envs.make("Solow-%s-%s-finite-eval-v0" % (p, q))


for (p, q) in [(z, z) for z in [1, 2, 3]]:
    register_solow_env(p, q)
    env = make_env(p, q)
    s_max = 0
    max_mean = 0
    stats = None
    for s in np.linspace(0.05, 0.95, 20):
        env.reset()
        done = False
        total_rewards = []
        while not done:
            _, reward, done, _ = env.step(s)
            total_rewards.append(reward)

        mean_rewards_s = np.mean(total_rewards)
        if mean_rewards_s > max_mean:
            max_mean = mean_rewards_s
            s_max = s
            stats = np.max(total_rewards), np.min(total_rewards), np.std(total_rewards)

    print(p, s_max, max_mean, stats)
