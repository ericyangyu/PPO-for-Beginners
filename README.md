# PPO for Beginners

## Introduction
Hi! My name is Eric Yu, and I wrote this repository to help beginners get started in writing Proximal Policy Optimization (PPO) from scratch using PyTorch. My goal is to provide a code for PPO that's bare-bones (little/no fancy tricks) and extremely well documented/styled and structured. I'm especially targeting people who are tired of reading endless PPO implementations and having absolutely no idea what's going on. 

If you're not coming from Medium, please read my article first: TODO

I wrote this code with the assumption that you have some experience with Python and Reinforcement Learning (RL), including how policy gradient (pg) algorithms and PPO work (for PPO, should just be familiar with theoretical level. After all, this code should help you with putting PPO into practice). If unfamiliar with RL, pg, or PPO, follow the three links below in order: <br />

If unfamiliar with RL, read [OpenAI Introduction to RL (all 3 parts)](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) <br />
If unfamiliar with pg, read [An Intuitive Explanation of Policy Gradient](https://towardsdatascience.com/an-intuitive-explanation-of-policy-gradient-part-1-reinforce-aa4392cbfd3c) <br />
If unfamiliar with PPO theory, read [PPO stack overflow post](https://stackoverflow.com/questions/46422845/what-is-the-way-to-understand-proximal-policy-optimization-algorithm-in-rl) <br />
If unfamiliar with all 3, go through those links above in order from top to bottom.

Please note that this PPO implementation assumes a continuous observation and action space, but you can change either to discrete relatively easily. I follow the pseudocode provided in OpenAI's Spinning Up for PPO: [https://spinningup.openai.com/en/latest/algorithms/ppo.html](https://spinningup.openai.com/en/latest/algorithms/ppo.html); pseudocode line numbers are specified as "ALG STEP #" in [ppo.py](./ppo.py).

Hope this is helpful, as I wish I had a resource like this when I started my journey into Reinforcement Learning.

## Usage
First I recommend creating a python virtual environment:
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

To train from scratch:
```
python main.py
```

To test model:
```
python main.py --mode test --actor_model ppo_actor.pth
```

To train with existing actor/critic models:
```
python main.py --actor_model ppo_actor.pth --critic_model ppo_critic.pth
```

NOTE: to change hyperparameters, environments, etc. do it in [main.py](main.py); I didn't have them as command line arguments because I don't like how long it makes the command.

## How it works

[main.py](main.py) is our executable. It will parse arguments using [arguments.py](arguments.py), then initialize our environment and PPO model. Depending on the mode you specify (train by default), it will train or test our model. To train our model, all we have to do is call ```learn``` function! This was designed with how you train PPO2 with [stable_baselines](https://stable-baselines.readthedocs.io/en/master/) in mind. 

[arguments.py](arguments.py) is what main will call to parse arguments from command line.

[ppo.py](ppo.py) contains our PPO model. All the learning magic happens in this file. Please read my Medium series (TODO) to see how it works. Another method I recommend is using something called ```pdb```, or python debugger, and stepping through my code starting from when I call ```learn``` in [main.py](main.py). 

[network.py](network.py) contains a sample Feed Forward Neural Network we can use to define our actor and critic networks in PPO. 

[eval_policy.py](eval_policy.py) contains the code to evaluating the policy. It's a completely separate module from the other code.

[graph_code directory](graph_code) contains the code to automatically collect data and generate graphs. Takes ~10 hours on a decent computer to generate all the data in my Medium article (TODO). All the data from the medium article should still be in ```graph_code/graph_data``` too in case you're interested; if you want, you can regenerate the graphs I use with the data. For more details, read the README in graph_code.

Here's a great pdb tutorial to get started: [https://www.youtube.com/watch?v=bHx8A8tbj2c](https://www.youtube.com/watch?v=bHx8A8tbj2c) <br />
Or if you're an expert with debuggers, here's the documentation: [https://docs.python.org/3/library/pdb.html](https://docs.python.org/3/library/pdb.html)

## Environments
Here's a [list of environments](https://github.com/openai/gym/wiki/Table-of-environments) you can try out. Note that in this PPO implementation, you can only use the ones with ```Box``` for both observation and action spaces.

I have no real recommendations for what values to set as hyperparameters for each environment, but feel free to play around with them. The usual suspects to change are the ```max_timesteps_per_episode``` and ```timesteps_per_batch``` hyperparameters; tune them according to what environment you select.

## Results

Please refer to my Medium article: TODO

## Contact

If you have any questions or would like to reach out to me, you can find me here: <br />
Email: eyyu@ucsd.edu <br />
LinkedIn: [https://www.linkedin.com/in/eric-yu-engineer/](https://www.linkedin.com/in/eric-yu-engineer/) <br />
