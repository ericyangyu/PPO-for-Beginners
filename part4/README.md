# Generating & Graphing Data

Here is where we can automatically generate data using PPO for Beginners and plot them onto graphs. Note that I use a lot more tricks in my code, making it a lot less readable for beginners, because I figured understanding how to automate data collection and graphing data doesn't align with the goal of PPO for Beginners. I did style and document the code though in case you wanted to take a closer look at how I generate and graph my data; just be warned that it will be hard to read and understanding the graph code isn't the goal of this repository and my [Medium series](https://medium.com/@eyyu/coding-ppo-from-scratch-with-pytorch-part-1-4-613dfc1b14c8). 

Also, the data in [graph_data](graph_data) should be the same data I used to make my graphs on my [Medium Part 1](https://medium.com/@eyyu/coding-ppo-from-scratch-with-pytorch-part-1-4-613dfc1b14c8), so you can just make my graphs again with the existing data (save yourself ~10 hours of training)

## Usage
To generate data:
```
./generate_data.bash <environment name 1> <environment name 2>
```
To graph the data generated:
```
python make_graph.py
```
Sample Use:
```
./generate_data.bash Pendulum-v0 BipedalWalker-v3
python make_graph.py
```
Note that you may need to change the permissions of generate_data.bash before using it, so you can do this:
```
chmod u+x generate_data.bash
```

## How It Works
[generate_data.bash](generate_data.bash) is how we generate data. It works by first taking in a list of environments to collect data on through command line arguments. Then, it calculates a set of random seeds per environment, and runs PPO for Beginners and Stable Baselines PPO2 on each seed on each environment until it covers all environments specified. You can change the configurations (like number of random seeds to test, range of seeds) in this file too. 

[make_graph.py](make_graph.py) is the code to plot the data stored in [graph_data](graph_data), and show pretty graphs on the screen for you to screenshot. In the graphs made, the thick lines are the averages over all the seeds and highlighted regions represent the variance.

[run.py](run.py) is called by [generate_data.bash](generate_data.bash), and is used to run the PPO for Beginners code or Stable Baselines PPO2 code with the right hyperparameters. 

[graph_data](graph_data) contains our data collected from [generate_data.bash](generate_data.bash). It's organized by:
1. environment
2. code (PPO for Beginners or Stable Baselines PPO2, also includes a seeds.txt containing all the seeds tested)
3. seed_xxx.txt (which is the output from training on this seed)

[PPO for Beginners](ppo_for_beginners) is just a copy of the PPO for Beginners code, so you can ignore this. It's used by [run.py](run.py) for convenience.
