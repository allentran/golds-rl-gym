import functools

import numpy as np
import gym


class SwarmEnv(gym.Env):
    N_LOCUSTS = 80 # number of locusts
    N_AGENTS = 10 # number of locust agents

    GRID_SIZE = 40

    # parameters of the system
    NOISE = 0.0001 # random noise on the locust velocities
    GRAVITY = -1 # gravity
    WIND_SPEED = 1 # wind speed
    F = 0.5 # attraction and repulsion parameters
    L = 10
    dt = 0.05 # time step for equations of motion

    N_BURN_IN = 10

    def __init__(self, seed=None) -> None:
        super().__init__()

        self.n_seed = seed
        self.states = None
        self.t = 0

    def _step(self, v_action, add_wind=True):
        x, xa = self.states

        v_action = v_action.copy()

        if add_wind:
            v_action[:, 0] += self.WIND_SPEED

        xa = self.x_update(xa, v_action, self.dt, self.NOISE * self.agent_noise[self.t]) # next state of robots
        v, reward = self.v_calculate(x, xa, self.F, self.L, self.WIND_SPEED, self.GRAVITY) # reward is energy, we want it to decrease
        x = self.x_update(x, v, self.dt, self.NOISE * self.particle_noise[self.t]) # next state of locusts

        self.states = [x, xa]

        return self.states, reward, reward >= 0, {}

    def _reset(self):
        if self.n_seed:
            np.random.seed(self.n_seed)
        self.t = 0

        x = np.random.rand(self.N_LOCUSTS, 2)
        xa = np.random.rand(self.N_AGENTS, 2)
        random_actions = np.random.normal(size=(self.N_BURN_IN, self.N_AGENTS, 2))

        self.agent_noise = np.random.normal(size=(128 + self.N_BURN_IN, self.N_AGENTS, 2))
        self.particle_noise = np.random.normal(size=(128 + self.N_BURN_IN, self.N_LOCUSTS, 2))
        self.states = [x, xa]

        for ii in range(self.N_BURN_IN):
            self.step(random_actions[ii])
            self.t += 1

        return self.states

    @staticmethod
    def s(r, F, L):
        s = F * np.exp(-r / L) - np.exp(-r)
        return s

    @staticmethod
    def x_update(x, v, dt, noise):
        x, v = SwarmEnv.xv_cutoff(x, v)
        x += dt * v + noise
        x, v = SwarmEnv.xv_cutoff(x, v)
        return x

    @staticmethod
    def xv_cutoff(x, v):
        N = v.shape[0]
        for j in range(N):
            if x[j][1] <= 0:
                x[j][1] = 0
                v[j][0] = 0
                if v[j][1] <= 0:
                    v[j][1] = 0
        return x, v

    @staticmethod
    def v_calculate(x, xa, F, L, U, G):
        N = x.shape[0]
        v = np.zeros((N,2))
        v[:, 0] = U
        v[:, 1] = G

        DISTS_AGENTS = np.sum((xa[:, :, None] - x.T[None, :, :]) ** 2, axis=1) ** 0.5
        DISTS_LOCUSTS = np.sum((x[:, :, None] - x.T[None, :, :]) ** 2, axis=1) ** 0.5

        dist_closest_locust = DISTS_AGENTS.min(axis=1).sum()

        for j in range(N):
            # other swarm particles
            dists = DISTS_LOCUSTS[:, j]
            v_0 = SwarmEnv.s(dists, F, L) * (x[:, 0] - x[j][0]) / (dists + 0.000001)
            v_1 = SwarmEnv.s(dists, F, L) * (x[:, 1] - x[j][1]) / (dists + 0.000001)
            v[j][0] += v_0.sum()
            v[j][1] += v_1.sum()

            # agent interactions
            dists = DISTS_AGENTS[:, j]
            v_0 = SwarmEnv.s(dists, F, L) * (xa[:, 0] - x[j][0]) / (dists + 0.000001)
            v_1 = SwarmEnv.s(dists, F, L) * (xa[:, 1] - x[j][1]) / (dists + 0.000001)
            v[j][0] += v_0.sum()
            v[j][1] += v_1.sum()
        energy = (v ** 2).sum(axis=1).mean()
        return v, -energy - 0.1 * dist_closest_locust

