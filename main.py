"""
	This file is the executable for running PPO. It is based on this medium article: TODO
"""

import gym
import sys
import torch

from arguments import get_args
from ppo import PPO
from network import FeedForwardNN

def train(model, actor_model, critic_model):
	"""
		Trains the model.

		Parameters:
			model - the model to use to train
			actor_model - the actor model to load in if we want to continue training
			critic_model - the critic model to load in if we want to continue training

		Return:
			None
	"""
	print(f"Training", flush=True)
	# Tries to load in an existing actor/critic model to continue training on
	if actor_model != '' and critic_model != '':
		print(f"Loading in {actor_model} and {critic_model}...", flush=True)
		model.actor.load_state_dict(torch.load(actor_model))
		model.critic.load_state_dict(torch.load(critic_model))
		print(f"Successfully loaded.", flush=True)
	elif actor_model != '' or critic_model != '': # Don't train from scratch if user accidentally forgets actor/critic model
		print(f"Error: Either specify both actor/critic models or none at all. We don't want to accidentally override anything!")
		sys.exit(0)
	else:
		print(f"Training from scratch.", flush=True)

	# Train the PPO model with a specified total timesteps
	model.learn(total_timesteps=200000000)

def test(model, actor_model):
	"""
		Tests the model.

		Parameters:
			model - the model to test with
			actor_model - the actor model to load in

		Return:
			None
	"""
	print(f"Testing {actor_model}", flush=True)
	# If the actor model is not specified, then exit
	if actor_model == '':
		print(f"Didn't specify model file. Exiting.", flush=True)
		sys.exit(0)

	# Load in the actor model
	model.actor.load_state_dict(torch.load(actor_model))

	# Render the environment
	model.render = True

	# Generate deterministic actions when rolling out
	model.deterministic = True

	# Autobots, roll out
	while True:
		model.rollout()
		model._log_summary()

def main(args):
	"""
		The main function to run.

		Parameters:
			args - the arguments parsed from command line

		Return:
			None
	"""
	# Creates the environment we'll be running. If you want to replace with your own
	# custom environment, note that it must inherit Gym and have both continuous
	# observation and action spaces.
	env = gym.make('Pendulum-v0')

	# Create a model for PPO. NOTE: Here's where you can set hyperparameters for PPO. I don't include them as part of
	# ArgumentParser because it's too annoying to type them every time at command line. Instead, you can change them here.
	# To see a list of hyperparameters, look in ppo.py at function _init_hyperparameters
	model = PPO(policy_class=FeedForwardNN, env=env, max_timesteps_per_episode=200, timesteps_per_batch=2000)

	# Train or test, depending on the mode specified
	if args.mode == 'train':
		train(model=model, actor_model=args.actor_model, critic_model=args.critic_model)
	else:
		test(model=model, actor_model=args.actor_model)


if __name__ == '__main__':
	args = get_args() # Parse arguments from command line
	main(args)
