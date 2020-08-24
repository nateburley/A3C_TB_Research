"""
This file will plot and analyze the results of training A3C (timesteps and rewards)
Use: python analyze_training_results.py /path/to/pkl/results plot_name

Author: Nathaniel M. Burley
Date: 15th August, 2020
"""
# Load modules
import pickle
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Pass in a file with the training rewards
try:
    results_file = sys.argv[1]
    plot_name = sys.argv[2]
except:
    print("Don't forget to pass in a file as a command line argument!")
    result_file = "results_test1/a3c/MsPacmanNoFrameskip_v4_08-07-2020/MsPacmanNoFrameskip_v4-a3c-rewards.pkl"
    plot_name = "plots/training_results_plot.png"

print("Results file: {}".format(results_file)) # Debugging


# Open the file for reading
with open(results_file, 'rb') as datafile:
    # Load the file; print the results (for debugging)
    rewards = pickle.load(datafile)

    # Load file into dataframe (training results)
    rewards_df = pd.DataFrame.from_dict(rewards['train'], orient='index', columns=['global_time', 'reward']).reset_index()
    print(rewards_df.head(10))

    # Get the rewards and timesteps
    steps = np.array(rewards_df.index)
    reward_values = np.array(rewards_df['reward'])
    
    # Plot the time steps and the rewards
    plt.plot(steps, reward_values, color='green', marker='o')
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Reward Value')
    plt.title(plot_name)
    plt.savefig('plots/' + plot_name + '-train-fig.png')

    ################## Do the same as above, but for the evaluation ################################
    rewards_eval_df = pd.DataFrame.from_dict(rewards['eval'], orient='index', columns=['global_time', 'reward']).reset_index()
    print(rewards_eval_df.head(10))

    # Get the rewards and timesteps
    eval_steps = np.array(rewards_eval_df.index)
    reward_eval_values = np.array(rewards_eval_df['reward'])
    
    # Plot the time steps and the rewards
    plt.plot(eval_steps, reward_eval_values, color='green', marker='o')
    plt.xlabel('Number of Steps')
    plt.ylabel('Reward Value')
    plt.title(plot_name)
    plt.savefig('plots/' + plot_name + '-eval-fig.png')

"""
Eval reward is the max score at each
- training reward is the average for one worker in its 20 steps (or until terminal)
- updating is done with one worker at a time-- adds how many steps it did and what rewards it got, basically

- transformed reward is computed in the training code, I just need to add more code to log it

- "KL divergence" between the two policies...distance between two distributions? (Could possibly be
used to compare the policies between two agents)
    - TB is for learning with high variance. Do NOT need to focus on this right now. 

- Good focus for this week would be a plot of raw reward, and a plot of transformed rewards, so we
can see if there's a correlation
"""