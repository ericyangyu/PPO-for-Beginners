"""
	This file will run Stable Baselines PPO2 or PPO for Beginners code
	with the input seed and environment.
"""

import gymnasium as gym
import os
import argparse

def train_stable_baselines(args):
	"""
		Trains with PPO2 on specified environment.

		Parameters:
			args - the arguments defined in main.

		Return:
			None
	"""
	from stable_baselines3 import PPO

	# Store hyperparameters and total timesteps to run by environment
	hyperparameters = {}
	total_timesteps = 0
	if args.env == 'Pendulum-v1':
		hyperparameters = {'n_steps': 2048, 'batch_size': 32, 'gae_lambda': 0.95, 'gamma': 0.99, 'n_epochs': 10,
							'ent_coef': 0.0, 'learning_rate': 3e-4, 'clip_range': 0.2, 'verbose': 1, 'seed': args.seed}
		total_timesteps = 1005000
	elif args.env == 'BipedalWalker-v3':
		hyperparameters = {'n_steps': 2048, 'batch_size': 32, 'gae_lambda': 0.95, 'gamma': 0.99, 'n_epochs': 10,
							'ent_coef': 0.001, 'learning_rate': 2.5e-4, 'clip_range': 0.2, 'verbose': 1, 'seed': args.seed}
		total_timesteps = 1405000
	elif args.env == 'LunarLanderContinuous-v2':
		hyperparameters = {'n_steps': 1024, 'batch_size': 32, 'gae_lambda': 0.98, 'gamma': 0.999, 'n_epochs': 4,
							'ent_coef': 0.01, 'clip_range': 0.2, 'verbose': 1, 'seed': args.seed}
		total_timesteps = 1005000
	elif args.env == 'MountainCarContinuous-v0':
		hyperparameters = {'n_steps': 256, 'batch_size': 8, 'gae_lambda': 0.94, 'gamma': 0.99, 'n_epochs': 4,
							'ent_coef': 0.0, 'clip_range': 0.2, 'verbose': 1, 'seed': args.seed}
		total_timesteps = 405000

	# Create log dir
	log_dir = "/tmp/gym/"
	os.makedirs(log_dir, exist_ok=True)

	# Make the environment and model, and train
	env = gym.make(args.env)
	model = PPO("MlpPolicy", env, **hyperparameters)
	model.learn(total_timesteps)

def train_ppo_for_beginners(args):
	"""
		Trains with PPO for Beginners on specified environment.

		Parameters:
			args - the arguments defined in main.

		Return:
			None
	"""
	# Import ppo for beginners
	if args.optimized:
		from ppo_for_beginners.ppo_optimized import PPO
	else:
		from ppo_for_beginners.ppo import PPO

	from ppo_for_beginners.network import FeedForwardNN

	# Store hyperparameters and total timesteps to run by environment
	hyperparameters = {}
	total_timesteps = 0
	if args.env == 'Pendulum-v1':
		hyperparameters = {'timesteps_per_batch': 2048, 'max_timesteps_per_episode': 200, 'gamma': 0.99, 'n_updates_per_iteration': 10,
							'lr': 3e-4,'lam': 0.98, 'clip': 0.2, 'save_freq': 1e15, 'seed': args.seed,'max_grad_norm':0.5,
							'target_kl':0.02,'ent_coef':0,'num_minibatches':6}
		total_timesteps = 1005000
	elif args.env == 'BipedalWalker-v3':
		hyperparameters = {'timesteps_per_batch': 2048, 'max_timesteps_per_episode': 1600, 'gamma': 0.99, 'n_updates_per_iteration': 10,
							'lr': 2.5e-4,'lam': 0.98, 'clip': 0.2, 'save_freq': 1e15, 'seed': args.seed,'max_grad_norm':0.5,
							'target_kl':0.02,'ent_coef':0,'num_minibatches':6}
		total_timesteps = 1405000
	elif args.env == 'LunarLanderContinuous-v2':
		hyperparameters = {'timesteps_per_batch': 1024, 'max_timesteps_per_episode': 1000, 'gamma': 0.999, 'n_updates_per_iteration': 4,
							'lr': 2.5e-4,'lam': 0.98, 'clip': 0.2, 'save_freq': 1e15, 'seed': args.seed,'max_grad_norm':0.5,
							'target_kl':0.02,'ent_coef':0,'num_minibatches':6}
		total_timesteps = 1005000
	elif args.env == 'MountainCarContinuous-v0':
		hyperparameters = {'timesteps_per_batch': 256, 'max_timesteps_per_episode': 1000, 'gamma': 0.99, 'n_updates_per_iteration': 4,
							'lr': 5e-3, 'lam': 0.98,'clip': 0.2, 'save_freq': 1e15, 'seed': args.seed,'max_grad_norm':0.5,
							'target_kl':0.02,'ent_coef':0,'num_minibatches':6}
		total_timesteps = 405000

	# Make the environment and model, and train
	env = gym.make(args.env)
	model = PPO(FeedForwardNN, env, **hyperparameters)
	model.learn(total_timesteps)

def main(args):
	"""
		An intermediate function that will call either PPO2 learn or PPO for Beginners learn.

		Parameters:
			args - the arguments defined below

		Return:
			None
	"""
	if args.code == 'stable_baselines_ppo':
		train_stable_baselines(args)
	elif args.code == 'ppo_for_beginners':
		train_ppo_for_beginners(args)

if __name__ == '__main__':
	# Parse arguments
	parser = argparse.ArgumentParser()

	parser.add_argument('--t', dest='optimized', type=bool, default=False)         # A flag for optimization
	parser.add_argument('--code', dest='code', type=str, default='')               # Can be 'stable_baselines_ppo' or 'ppo_for_beginners' / 'ppo_for_beginners/trick'
	parser.add_argument('--seed', dest='seed', type=int, default=None)             # An int for our seed
	parser.add_argument('--env', dest='env', type=str, default='')                 # Formal name of environment

	args = parser.parse_args()

	# Collect data
	main(args)