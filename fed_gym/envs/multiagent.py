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

    def __init__(self) -> None:
        super().__init__()

        self.states = None

    def _step(self, v_action, add_wind=True):
        x, xa = self.states

        if add_wind:
            v_action[:, 0] += self.WIND_SPEED

        xa = self.x_update(xa, v_action, self.dt, self.NOISE) # next state of robots
        v, reward = self.v_calculate(x, xa, self.F, self.L, self.WIND_SPEED, self.GRAVITY) # reward is energy, we want it to decrease
        x = self.x_update(x, v, self.dt, self.NOISE) # next state of locusts

        self.states = [x, xa]

        return self.states, reward, reward >= 0, {}

    def _reset(self):
        x = np.random.rand(self.N_LOCUSTS, 2)
        xa = np.random.rand(self.N_AGENTS, 2)
        self.states = [x, xa]
        return self.states

    @staticmethod
    @functools.lru_cache()
    def s(r, F, L):
        s = F * np.exp(-r / L) - np.exp(-r)
        return s

    @staticmethod
    def x_update(x, v, dt, noise):
        N = v.shape[0]
        x, v = SwarmEnv.xv_cutoff(x, v)
        x += dt * v + noise * np.random.randn(N, 2)
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
        Na = xa.shape[0]
        v = np.zeros((N,2))
        v[:, 0] = U
        v[:, 1] = G
        for j in range(N):
            for k in range(N):
                if k != j:
                    dist = ((x[k][0] - x[j][0]) ** 2 + (x[k][1] - x[j][1]) ** 2) ** 0.5
                    v[j][0] += SwarmEnv.s(dist, F, L) * (x[k][0] - x[j][0]) / (dist + 0.000001)
                    v[j][1] += SwarmEnv.s(dist, F, L) * (x[k][1] - x[j][1]) / (dist + 0.000001)
            for k in range(Na):
                dist = ((xa[k][0] - x[j][0]) ** 2 + (xa[k][1] - x[j][1]) ** 2) ** 0.5
                v[j][0] += SwarmEnv.s(dist, F, L) * (xa[k][0] - x[j][0]) / (dist + 0.000001)
                v[j][1] += SwarmEnv.s(dist, F, L) * (xa[k][1] - x[j][1]) / (dist + 0.000001)
        energy = (v ** 2).sum()
        return v, -energy

