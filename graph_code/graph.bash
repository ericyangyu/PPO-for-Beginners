#!/bin/bash

# Usage: ./graph.bash env1 env2 env3
#
# Example: ./graph.bash Pendulum-v0 BipedalWalker-v3

# Generate random seeds
SEEDS=($(shuf -i 0-1000 -n 5))

# Create logging directories if missing
mkdir -p graph_data/stable_baselines
mkdir -p graph_data/ppo_for_beginners

# Store the seeds into our graph_data directory
echo -n > graph_data/seeds.txt
for seed in "${SEEDS[@]}"; do
    echo "$seed" >> graph_data/seeds.txt
done

# Iterate through all argument environments passed in
for env in "$@"
do
	# Create logging directory for specified environment
	# Note that we're clearing the environment directory first to only retain most recent run data.
	# If you have data in those env directories that you don't want to lose, move them elsewhere!!
	rm -rf graph_data/stable_baselines/$env
	rm -rf graph_data/ppo_for_beginners/$env
	mkdir -p graph_data/stable_baselines/$env
	mkdir -p graph_data/ppo_for_beginners/$env

	# Loop through each seed
	for seed in "${SEEDS[@]}"; do
		python run.py --code 'stable_baselines_ppo2' --env $env --seed $seed > "./graph_data/stable_baselines/$env/seed_$seed.txt"
		python run.py --code 'ppo_for_beginners' --env $env --seed $seed > "./graph_data/ppo_for_beginners/$env/seed_$seed.txt"
	done
done

# Done
echo "Successfully generated all data!"
