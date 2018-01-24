import json

import numpy as np
import gym

from fed_gym.envs.fed_env import register_solow_env



def make_env(p, q):
    return gym.envs.make("Solow-%s-%s-finite-v0" % (p, q))

N = 100

for (p, q) in [(z, z) for z in [1]]:
    register_solow_env(p, q)
    env = make_env(p, q)
    s_max = 0
    max_mean = 0
    stats = None
    for s in np.linspace(0.05, 0.95, 20):
        mean_rewards = []
        for n in range(N):
            env.reset()
            done = False
            total_rewards = []
            while not done:
                _, reward, done, _ = env.step(s)
                total_rewards.append(reward)
            mean_rewards.append(np.mean(total_rewards))

        mean_rewards_s = np.mean(mean_rewards)
        if mean_rewards_s > max_mean:
            max_mean = mean_rewards_s
            s_max = s
            stats = np.max(mean_rewards), np.min(mean_rewards), np.std(mean_rewards)

    print(p, s_max, max_mean, stats)
