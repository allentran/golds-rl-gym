from gym.envs.registration import register

register(
    id='fed-v0',
    entry_point='fed_gym.envs:TradeEnv',
)
