[![CircleCI](https://circleci.com/gh/allentran/golds-rl-gym/tree/master.svg?style=svg)](https://circleci.com/gh/allentran/golds-rl-gym/tree/master)
# Environments + agents for Open AI Gym
* Solving partial state/multi-agent control problems with RL
* Implementations of continuous control PAAC and A3C
* Finance/trading environments

## Swarm environment
Learning to control a swarm of locusts (realistic physics)

Random | After 24 hours
------------ | -------------
![Random](https://media.giphy.com/media/jnVo0Fe0cBqAJgS3Ja/giphy.gif) | ![Learned](https://media.giphy.com/media/EfmXvtNq3ZnjHTAY2Z/giphy.gif)

To train
```
python scripts/train_paac_conv.py -d /gpu:0 --height=84 --clip_norm=1
```
