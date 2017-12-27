from gym.envs.registration import register

register(
    id='Fed-v0',
    entry_point='fed_gym.envs:TradeEnv',
)

register(
    id='Solow-v0',
    entry_point='fed_gym.envs:SolowEnv',
)
