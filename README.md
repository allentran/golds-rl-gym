[![CircleCI](https://circleci.com/gh/allentran/golds-rl-gym/tree/master.svg?style=svg)](https://circleci.com/gh/allentran/golds-rl-gym/tree/master)
# Environments + agents for Open AI Gym
* Solving partial state/multi-agent control problems with RL
* Implementations of continuous control PAAC and A3C
* Finance/trading environments

## Swarm environment
Learning to stop a swarm of locusts from "rolling"

Random | After 24 hours
------------ | -------------
![Random](https://media.giphy.com/media/xtpkwoUKvljrcvMGqJ/giphy.gif) | ![Learned](https://media.giphy.com/media/g0w05sZFV8xhve9rzE/giphy.gif)

To train
```
python scripts/train_paac_conv.py -d /gpu:0 --height=84 --clip_norm=1
```
