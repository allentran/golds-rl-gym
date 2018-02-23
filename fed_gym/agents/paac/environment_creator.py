import gym


class SolowEnvironmentCreator(object):

    def __init__(self, p, q):
        """
        Creates an object from which new environments can be created
        :param args:
        """

        self.num_actions = 1
        self.create_environment = lambda: gym.envs.make("Solow-%s-%s-finite-v0" % (p, q))


class SwarmEnvironmentCreator(object):
    def __init__(self) -> None:
        super().__init__()
        self.num_actions = 2
        self.create_environment = lambda: gym.envs.make("Swarm-v0")
