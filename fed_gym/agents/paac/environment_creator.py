import gym


class EnvironmentCreator(object):

    def __init__(self, p, q):
        """
        Creates an object from which new environments can be created
        :param args:
        """

        self.num_actions = 1
        self.create_environment = lambda: gym.envs.make("Solow-%s-%s-finite-v0" % (p, q))



