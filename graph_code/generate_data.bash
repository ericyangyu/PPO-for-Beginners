#!/bin/bash

# This file generates data on the specified command line argument environments

# CHANGE ME: These are the specifications for generating data
NUM_SEEDS=5                            # Number of random seeds to test per environment
SEED_LOWER_BOUND=0                     # Lower bound of random seeds
SEED_UPPER_BOUND=1000                  # Upper bound of random seeds

# Generate random seeds
SEEDS=($(shuf -i $SEED_LOWER_BOUND-$SEED_UPPER_BOUND -n $NUM_SEEDS))
# Or use fixed seeds
# SEEDS=(495 741 585 263 325) # Pendulum
# SEEDS=(702 742 298 470 797) # Bipedal
# SEEDS=(141 957 110 875 120) # Lunarlander-v2 
# Or use any seeds you want

# Create graph_data directory if missing
mkdir -p graph_data
optimized_flag=false
# Iterate through all argument environments passed in
for env in "$@"
do
	# Check for optimization flag
	if [ "$env" == "-t" ]; then
        optimized_flag=true
		continue
    fi
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

	for seed in "${SEEDS[@]}"; do
		python run.py --t true --code 'stable_baselines_ppo' --env $env --seed $seed > "./graph_data/$env/stable_baselines/seed_$seed.txt"
		python run.py --t $optimized_flag --code 'ppo_for_beginners' --env $env --seed $seed > "./graph_data/$env/ppo_for_beginners/seed_$seed.txt"
	done

done

# Done
echo "Successfully generated all data. The data are stored in graph_data directory. To generate graphs, run make_graph.py"
