from gym.envs.registration import register

register(
    id='TradeAR1-v0',
    entry_point='fed_gym.envs:TradeAR1Env',
    max_episode_steps=1024
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

register(
    id='Swarm-v0',
    entry_point='fed_gym.envs:SwarmEnv',
)

