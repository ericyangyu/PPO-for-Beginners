"""
    This file graphs the data in graph_data for each environment
"""
import os
import numpy as np
import matplotlib.pyplot as plt

def get_file_locations():
    """
        Gets the absolute paths of each data file to graph.

        Parameters:
            None

        Return:
            paths - a dict with the following structure:
            {
                env: {
                seeds: absolute_seeds_file_path
                stable_baselines: [absolute_file_paths_to_data],
                ppo_for_beginners: [absolute_file_paths_to_data]
                }
            }
    """
    # Get the absolute path of current working directory, 
    # and append graph data to it
    data_path = os.getcwd() + '/graph_data/'

    # extract environments from envs
    envs = [env for env in os.listdir(data_path) 
                        if os.path.isdir(os.path.join(data_path,env))]

    # Stores the file names.
    # Structure will be:
    # {
    #   env: {
    #            seeds: absolute_seeds_file_path
    #            stable_baselines: [absolute_file_paths_to_data],
    #            ppo_for_beginners: [absolute_file_paths_to_data]
    #        }
    # }
    paths = {}

    for env in envs:
        # Sub-dict in env as listed above
        env_data = {}
        # Extract out absolute paths of seeds.txt and data for both codes
        for directory, _, filenames in os.walk(data_path + env):
            if 'seeds.txt' in filenames:
                env_data['seeds'] = directory + '/' + filenames[0]
            elif 'stable_baselines' in directory:
                env_data['stable_baselines'] = [directory + '/' + filename 
                                                    for filename in filenames]
            elif 'ppo_for_beginners' in directory:
                env_data['ppo_for_beginners'] = [directory + '/' + filename 
                                                    for filename in filenames]

        # save the environment data into our outer paths dict
        paths[env] = env_data

    return paths

def extract_ppo_for_beginners_data(env, filename):
    """
        Extract the total timesteps and average episodic return from the logging
        data specified to PPO for Beginners.

        Parameters:
            env - The environment we're currently graphing.
            filename - The file containing data. Should be "seed_xxx.txt" such that the
                        xxx are integers

        Return:
            x - the total timesteps at each iteration
            y - average episodic return at each iteration
    """
    # x is timesteps so far, y is average episodic reward
    x, y = [], []

    # extract out x's and y's
    with open(filename, 'r') as f:
        for l in f:
            l = [e.strip() for e in l.split(':')]
            if 'Average Episodic Return' in l:
                y.append(float(l[1]))
            if 'Timesteps So Far' in l:
                x.append(int(l[1]))

    return x, y

def extract_stable_baselines_data(env, filename):
    """
        Extract the total timesteps and average episodic return from the logging
        data specified to Stable Baselines PPO2.

        Parameters:
            env - The environment we're currently graphing.
            filename - The file containing data. Should be "seed_xxx.txt" such that the
                        xxx are integers

        Return:
            x - the total timesteps at each iteration
            y - average episodic return at each iteration
    """
    # x is timesteps so far, y is average episodic reward
    x, y = [], []

    # extract out x's and y's
    with open(filename, 'r') as f:
        for l in f:
            l = [e.strip() for e in l.split('|')]
            if 'ep_reward_mean' in l:
                y.append(float(l[2]))
            if 'total_timesteps' in l:
                x.append(int(l[2]))
    return x, y

def calculate_lower_bounds(x_s, y_s):
    """
        Calculate lower bounds of total timesteps and average episodic
        return per iteration.

        Parameters:
            x_s - A list of lists of total timesteps so far per seed.
            y_s - A list of lists of average episodic return per seed.

        Return: 
            Lower bounds of both x_s and y_s
    """
    # x_low is lower bound of timesteps so far, y is lower bound of average episodic reward
    x_low, y_low = x_s[0], y_s[0]

    # Find lower bound amongst all trials per iteration
    for xs, ys in zip(x_s[1:], y_s[1:]):
        x_low = [x if x < x_low[i] else x_low[i] for i, x in enumerate(xs)]
        y_low = [y if y < y_low[i] else y_low[i] for i, y in enumerate(ys)]
    return x_low, y_low

def calculate_upper_bounds(x_s, y_s):
    """
        Calculate upper bounds of total timesteps and average episodic
        return per iteration.

        Parameters:
            x_s - A list of lists of total timesteps so far per seed.
            y_s - A list of lists of average episodic return per seed.

        Return: 
            Upper bounds of both x_s and y_s
    """
    # x_low is upper bound of timesteps so far, y is upper bound of average episodic reward
    x_high, y_high = x_s[0], y_s[0]

    # Find upper bound amongst all trials per iteration
    for xs, ys in zip(x_s[1:], y_s[1:]):
        x_high = [x if x > x_high[i] else x_high[i] for i, x in enumerate(xs)]
        y_high = [y if y > y_high[i] else y_high[i] for i, y in enumerate(ys)]
    return x_high, y_high

def calculate_means(x_s, y_s):
    """
        Calculate mean of each total timestep and average episodic return over all
        trials at each iteration.

        Parameters:
            x_s - A list of lists of total timesteps so far per seed.
            y_s - A list of lists of average episodic return per seed

        Return: 
            Means of x_s and y_s 
    """
    if len(x_s) == 1:
        return x_s, y_s

    return list(np.mean(x_s, axis=0)), list(np.mean(y_s, axis=0))

def clip_data(x_s, y_s):
    """
        In the case that there are different number of iterations
        across learning trials, clip all trials to the length of the shortest
        trial.

        Parameters:
            x_s - A list of lists of total timesteps so far per seed.
            y_s - A list of lists of average episodic return per seed

        Return: 
            x_s and y_s after clipping both. 
    """
    # Find shortest trial length
    x_len_min = min([len(x) for x in x_s])
    y_len_min = min([len(y) for y in y_s])

    len_min = min([x_len_min, y_len_min])

    # Clip each trial in x_s to shortest trial length
    for i in range(len(x_s)):
        x_s[i] = x_s[i][:len_min]

    # Clip each trial in y_s to shortest trial length
    for i in range(len(y_s)):
        y_s[i] = y_s[i][:len_min]
       
    return x_s, y_s

def extract_data(paths):
    """
        Extracts data from all the files, and returns a generator object
        extract_data to iterably return data for each environment. 
        Number of iterations should equal number of environments in graph_data.

        Parameters:
            paths - Contains the paths to each data file. Check function description of 
                    get_file_locations() to see how paths is structured. 

        Return: 
            A generator object extract_data, or iterable, which will return the data for
			each environment on each iteration of the generator.

		Note:
			If you're unfamiliar with Python generators, check this out:
				https://wiki.python.org/moin/Generators
			If you're unfamiliar with Python "yield", check this out:
				https://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do
    """
    for env in paths:
        # Extract out seeds tested
        seeds_txt = paths[env]['seeds']
        seeds = []
        with open(seeds_txt, 'r') as f:
            for l in f:
                seeds.append(int(l))

        # Prepare the data dict to return
        data = {
            'env': '',
            'seeds': [],
            'ppo_for_beginners': {
                'x_mean': [],
                'x_low': [],
                'x_high': [],
                'y_mean': [],
                'y_low': [],
                'y_high': []
            },
            'stable_baselines': {
                'x_mean': [],
                'x_low': [],
                'x_high': [],
                'y_mean': [],
                'y_low': [],
                'y_high': []
            }
        }

        # Extract out ppo_beginner datapoints
        pfb_x_s, pfb_y_s = [], []
        for filename in paths[env]['ppo_for_beginners']:
            curr_data = extract_ppo_for_beginners_data(env, filename)
            pfb_x_s.append(curr_data[0])
            pfb_y_s.append(curr_data[1])

        # Extract out stable_baselines datapoints
        sb_x_s, sb_y_s = [], []
        for filename in paths[env]['stable_baselines']:
            curr_data = extract_stable_baselines_data(env, filename)
            sb_x_s.append(curr_data[0])
            sb_y_s.append(curr_data[1])

        # Preprocess ppo_beginner and stable_baselines data
        pfb_x_s, pfb_y_s = clip_data(pfb_x_s, pfb_y_s)
        sb_x_s, sb_y_s = clip_data(sb_x_s, sb_y_s)

        # Process ppo_beginner datapoints for mean, lower, and upper bounds
        pfb_x_mean, pfb_y_mean = calculate_means(pfb_x_s, pfb_y_s)
        pfb_x_low, pfb_y_low = calculate_lower_bounds(pfb_x_s, pfb_y_s)
        pfb_x_high, pfb_y_high = calculate_upper_bounds(pfb_x_s, pfb_y_s)

        # Process stable_baselines datapoints for mean, lower, and upper bounds
        sb_x_mean, sb_y_mean = calculate_means(sb_x_s, sb_y_s)
        sb_x_low, sb_y_low = calculate_lower_bounds(sb_x_s, sb_y_s)
        sb_x_high, sb_y_high = calculate_upper_bounds(sb_x_s, sb_y_s)

        # Intermediary variables to help us more easily save our data
        pfbs = [pfb_x_mean, pfb_x_low, pfb_x_high, pfb_y_mean, pfb_y_low, pfb_y_high]
        sbs = [sb_x_mean, sb_x_low, sb_x_high, sb_y_mean, sb_y_low, sb_y_high]

        # Fill up data packet
        data['env'] = env
        data['seeds'] = seeds
        for i, data_type in enumerate(['x_mean', 'x_low', 'x_high', 'y_mean', 'y_low', 'y_high']):
            data['ppo_for_beginners'][data_type] = pfbs[i]
            data['stable_baselines'][data_type] = sbs[i]

        # Return current data packet
        yield data

def graph_data(paths):
    """
        Graphs the data with matplotlib. Will display on screen for user to screenshot.

        Parameters:
            paths - Contains the paths to each data file. Check function description of 
                    get_file_locations() to see how paths is structured. 

        Return:
            None
    """
    for data in extract_data(paths):
        # Unpack data packet
        env = data['env']
        seeds = data['seeds']

        pfbs = [pfb_x_mean, _, _, pfb_y_mean, pfb_y_low, pfb_y_high] = [[] for _ in range(6)]
        sbs = [sb_x_mean, _, _, sb_y_mean, sb_y_low, sb_y_high] = [[] for _ in range(6)]
        for i, data_type in enumerate(['x_mean', 'x_low', 'x_high', 'y_mean', 'y_low', 'y_high']):
            pfbs[i] = data['ppo_for_beginners'][data_type]
            sbs[i] = data['stable_baselines'][data_type]
        
        pfb_x_mean, _, _, pfb_y_mean, pfb_y_low, pfb_y_high = pfbs
        sb_x_mean, _, _, sb_y_mean, sb_y_low, sb_y_high = sbs

        # Handle specific case with mountaincarcontinuous
        if env == 'MountainCarContinuous-v0':
            plt.ylim([-70, 100])

        # Plot points
        plt.plot(sb_x_mean, sb_y_mean, 'b', alpha=0.8)
        plt.plot(pfb_x_mean, pfb_y_mean, 'g', alpha=0.8)
        # Plot errors
        plt.fill_between(sb_x_mean, sb_y_low, sb_y_high, color='b', alpha=0.3)
        plt.fill_between(pfb_x_mean, pfb_y_low, pfb_y_high, color='g', alpha=0.3)
        # Set labels
        plt.title(f'{env} on Random Seeds {seeds}')
        plt.xlabel('Average Total Timesteps So Far')
        plt.ylabel('Average Episodic Return')
        plt.legend(['Stable Baselines PPO2', 'PPO for Beginners'])
        # Show graph so user can screenshot
        plt.show()

def main():
    """
        Main function to get file locations and graph the data.

        Parameters:
            None

        Return:
            None
    """
    # Extract absolute file paths
    paths = get_file_locations()

    # Graph the data from the file paths extracted
    graph_data(paths)

if __name__ == '__main__':
	main()