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
def explore_row_log(row_log_pkl):
    with open(row_log_pkl, 'rb') as datafile:
        # Read in the cumsum rewards
        raw_returns = []
        rewards = []
        tb_returns = []
        while True:
            try:
                batch = pickle.load(datafile)
                for reward, raw_return, batch_return in zip(batch.rewards, batch.batch_raw_rewards, batch.batch_cumsum_rewards):
                    rewards.append(reward)
                    raw_returns.append(raw_return)
                    tb_returns.append(batch_return)
            except EOFError:
                break
        
        # Plot a histogram (bins=[0, 0.5, 1, 1.5, 2, 4, 6, 8, 10, 12, 14, 16])
        plt.hist(rewards, color="blue", label="Rewards", alpha=0.5, bins=[0, 0.5, 1, 1.5, 2, 4, 6, 8, 10, 12, 14, 16])
        plt.hist(raw_returns, color="yellow", label="Raw Batch Returns", alpha=0.5, bins=[0, 0.5, 1, 1.5, 2, 4, 6, 8, 10, 12, 14, 16])
        plt.hist(tb_returns, color="red", label="TB Batch Returns", alpha=0.5, bins=[0, 0.5, 1, 1.5, 2, 4, 6, 8, 10, 12, 14, 16])
        plt.title("Number of Occurrences of Rewards, Batch Cumulative Rewards, and Raw Rewards [TB]")
        plt.xlabel("Value")
        plt.ylabel("Number of Occurrences")
        plt.legend()
        plt.show()
        

explore_row_log('results/RowLogs/TB_ROW-LOG.pkl')