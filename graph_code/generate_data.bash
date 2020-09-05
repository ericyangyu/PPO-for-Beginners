#!/bin/bash

# Usage: ./generate_data.bash env1 env2 env3
#
# Example: ./generate_data.bash Pendulum-v0 BipedalWalker-v3

# CHANGE ME: These are the specifications for generating data
NUM_SEEDS=5
SEED_LOWER_BOUND=0
SEED_UPPER_BOUND=1000

# Generate random seeds
SEEDS=($(shuf -i $SEED_LOWER_BOUND-$SEED_UPPER_BOUND -n $NUM_SEEDS))

# Create logging directories if missing
mkdir -p graph_data
mkdir -p graph_data

# Iterate through all argument environments passed in
for env in "$@"
do
	# Create logging directory for specified environment
	# Note that we're clearing the environment directory first to only retain most recent run data.
	# If you have data in those env directories that you don't want to lose, move them elsewhere!!
	rm -rf graph_data/$env/
	mkdir -p graph_data/$env/stable_baselines
	mkdir -p graph_data/$env/ppo_for_beginners

	# Store the seeds into our graph_data directory
	echo -n > graph_data/$env/seeds.txt
	for seed in "${SEEDS[@]}"; do
		echo "$seed" >> graph_data/$env/seeds.txt
	done

	# Loop through each seed and train
	for seed in "${SEEDS[@]}"; do
		python run.py --code 'stable_baselines_ppo2' --env $env --seed $seed > "./graph_data/$env/stable_baselines/seed_$seed.txt"
		python run.py --code 'ppo_for_beginners' --env $env --seed $seed > "./graph_data/$env/ppo_for_beginners/seed_$seed.txt"
	done
done

# Done
echo "Successfully generated all data. The data are stored in graph_data directory. To generate graphs, run make_graph.py"
