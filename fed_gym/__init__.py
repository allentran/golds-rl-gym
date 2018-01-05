from gym.envs.registration import register

register(
    id='Fed-v0',
    entry_point='fed_gym.envs:TradeEnv',
)

register(
    id='Solow-v0',
    entry_point='fed_gym.envs:SolowEnv',
    max_episode_steps=1024
)

register(
    id='SolowSS-v0',
    entry_point='fed_gym.envs:SolowSSEnv',
    max_episode_steps=1024
)
