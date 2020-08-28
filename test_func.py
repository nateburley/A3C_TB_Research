import numpy as np
import pickle
import matplotlib.pyplot as plt

# Function used in the Bellman Transform (for rewards)
def transform_h(z, eps=10**-2):
    return (np.sign(z) * (np.sqrt(np.abs(z) + 1.) - 1.)) + (eps * z)

# Function that computes the inverse of the above (used in Bellman Transform)
def transform_h_inv(z, eps=10**-2):
    return np.sign(z) * (np.square((np.sqrt(1 + 4 * eps * (np.abs(z) + 1 + eps)) - 1) / (2 * eps)) - 1)

# print("Original score: {}\n Transformed score: {}".format(1, transform_h(1 + 0.99 * transform_h_inv(1))))

# Function that tests my cumsum reward logging, plots histogram (for Yunshu's exercise)
def explore_cumsum(cumsum_log_pkl):
    with open(cumsum_log_pkl, 'rb') as datafile:
        # Read in the cumsum rewards
        rewards_list = []
        while True:
            try:
                batch_rewards = pickle.load(datafile) # Length is 20? Not sure what a "batch" is...
                for x in batch_rewards: rewards_list.append(x)
            except EOFError:
                break
        
        # Plot a histogram
        plt.hist(rewards_list, bins=[0, 0.5, 1, 1.5, 2, 4, 6, 8, 10, 12, 14, 16])
        plt.title("Number of Occurrences of Cum. Sum. Reward Values [A3C]")
        plt.xlabel("Reward Value")
        plt.ylabel("Number of Reward Occurrences")
        plt.show()
        

explore_cumsum('cumulative_rewards.pkl')