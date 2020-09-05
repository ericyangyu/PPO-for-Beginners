import os
import numpy as np
import matplotlib.pyplot as plt

def get_file_locations():
    # Get the absolute path of current working directory, 
    # and append graph data to it
    data_path = os.getcwd() + '/graph_data/'

    # extract environments from envs
    envs = [env for env in os.listdir(data_path) 
                        if os.path.isdir(os.path.join(data_path,env))]

    # Stores the file names.
    # Structure will be:
    # {env: {
    #            seeds: absolute_seeds_file_path
    #            stable_baselines: [absolute_file_paths_to_data],
    #            ppo_for_beginners: [absolute_file_paths_to_data]
    #       }
    # }
    paths = {}

    for env in envs:
        # Sub-dict in env as listed above
        env_data = {}
        # Extract out absolute paths of seeds.txt and data for both codes
        for directory, _, filenames in os.walk(data_path + env):
            if 'seeds.txt' in filenames:
                env_data['seeds'] = directory + '/' + filenames[0]
            elif 'ppo_for_beginners' in directory:
                env_data['ppo_for_beginners'] = [directory + '/' + filename 
                                                    for filename in filenames]
            elif 'stable_baselines' in directory:
                env_data['stable_baselines'] = [directory + '/' + filename 
                                                    for filename in filenames]

        # save the environment data into our outer paths dict
        paths[env] = env_data

    return paths

def extract_ppo_for_beginners_data(env, filename):
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
    x, y = [], []
    with open(filename, 'r') as f:
        for l in f:
            l = [e.strip() for e in l.split('|')]
            if 'ep_reward_mean' in l:
                y.append(float(l[2]))
            if 'total_timesteps' in l:
                x.append(int(l[2]))
    return x, y

def calculate_lower_bounds(x_s, y_s):
    x_low, y_low = x_s[0], y_s[0]
    for xs, ys in zip(x_s[1:], y_s[1:]):
        x_low = [x if x < x_low[i] else x_low[i] for i, x in enumerate(xs)]
        y_low = [y if y < y_low[i] else y_low[i] for i, y in enumerate(ys)]
    return x_low, y_low

def calculate_upper_bounds(x_s, y_s):
    x_high, y_high = x_s[0], y_s[0]
    for xs, ys in zip(x_s[1:], y_s[1:]):
        x_high = [x if x > x_high[i] else x_high[i] for i, x in enumerate(xs)]
        y_high = [y if y > y_high[i] else y_high[i] for i, y in enumerate(ys)]
    return x_high, y_high

def calculate_means(x_s, y_s):
    if len(x_s) == 1:
        return x_s, y_s

    return list(np.mean(x_s, axis=0)), list(np.mean(y_s, axis=0))

def clip_data(x_s, y_s):
    """
        In the case that there are different number of iterations
        across learning trials, clip all trials to the length of the smallest
    """
    x_len_min = min([len(x) for x in x_s])
    y_len_min = min([len(y) for y in y_s])

    len_min = min([x_len_min, y_len_min])

    for i in range(len(x_s)):
        x_s[i] = x_s[i][:len_min]

    for i in range(len(y_s)):
        y_s[i] = y_s[i][:len_min]
       
    return x_s, y_s


def extract_data(paths):
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

        # Yield current data packet
        yield data

def graph_data(paths):
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
        plt.plot(pfb_x_mean, pfb_y_mean, 'y', alpha=0.8)
        # Plot errors
        plt.fill_between(sb_x_mean, sb_y_low, sb_y_high, color='b', alpha=0.4)
        plt.fill_between(pfb_x_mean, pfb_y_low, pfb_y_high, color='y', alpha=0.4)
        # Set labels
        plt.title(f'{env} on Random Seeds {seeds}')
        plt.xlabel('Average Total Timesteps So Far')
        plt.ylabel('Average Episodic Return')
        # Show graph so user can screenshot
        plt.show()

def main():
    paths = get_file_locations()
    graph_data(paths)

if __name__ == '__main__':
	main()

# x_1 = []
# y_1 = []

# with open('stable_baselines_out', 'r') as f:
# 	for l in f:
# 		l = [e.strip() for e in l.split('|')]
# 		if 'ep_reward_mean' in l:
# 			y_1.append(float(l[2]))
# 		if 'total_timesteps' in l:
# 			x_1.append(int(l[2]))
# 			if int(l[2]) > 400000:
# 				break

# x_2 = []
# y_2 = []

# with open('my_out', 'r') as f:
# 	for l in f:
# 		l = [e.strip() for e in l.split(':')]
# 		if 'Average Episodic Return' in l:
# 			y_2.append(float(l[1]))
# 		if 'Timesteps So Far' in l:
# 			x_2.append(int(l[1]))
# 			if int(l[1]) > 400000:
# 				break

# plt.ylim([-30, 100])
# plt.plot(x_1, y_1, 'b', alpha=0.75)
# plt.plot(x_2, y_2, 'y', alpha=0.75)
# plt.title('MountainCarContinuous-v0')
# plt.legend(['stable_baselines PPO2', 'PPO for Beginners'])
# plt.xlabel('Total Timesteps So Far')
# plt.ylabel('Average Episodic Return')
# plt.show()