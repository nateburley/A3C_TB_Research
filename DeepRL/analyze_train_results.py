import numpy as np
import pickle
import matplotlib.pyplot as plt

# Function that tests my cumsum reward logging, plots histogram (for Yunshu's exercise)
# IMPORTANT NOTE: The actual training does 5 "trials" or whatever. If it writes all of those to the same file,
#                 then you need to add some logic that compares the global_t values to see when in the file one
#                 one trial ends and the next begins
def explore_row_log(row_log_pkl, TB=True):
    with open(row_log_pkl, 'rb') as datafile:

        # Plot the histogram(s) for A3C with the Transformed Bellman
        if (TB == True):
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
            
            # Plot a histogram of raw vs. TB returns
            plt.hist(raw_returns, color="blue", label="Raw Batch Returns", alpha=0.5, bins=15)
            plt.hist(tb_returns, color="yellow", label="TB Batch Returns", alpha=0.5, bins=15)
            plt.title("Number of Occurrences of TB Batch Cumulative Rewards vs. Raw Rewards [TB]")
            plt.xlabel("Value")
            plt.ylabel("Number of Occurrences")
            plt.legend()
            plt.savefig("plots/TB_raw-vs-trans_return_hist.png")

            # Clear the figure
            plt.clf()

            # Plot a histogram of rewards
            plt.hist(rewards, color="green", label="Rewards", bins=15)
            plt.title("Number of Occurrences of Rewards [TB]")
            plt.xlabel("Value")
            plt.ylabel("Number of Occurrences")
            plt.legend()
            plt.savefig("plots/TB_reward_hist.png")
        
        # Plot the histogram(s) for "normal" A3C
        else:
            # Read in the cumsum rewards
            cumsum_rewards = []
            rewards = []
            while True:
                try:
                    batch = pickle.load(datafile)
                    for reward, batch_return in zip(batch.rewards, batch.batch_cumsum_rewards):
                        rewards.append(reward)
                        cumsum_rewards.append(batch_return)
                except EOFError:
                    break

            # Plot historgram of returns
            plt.hist(cumsum_rewards, color="blue", label="Cum. Sum Rewards", bins=15)
            plt.title("Number of Occurrences of Cumulative Rewards [A3C]")
            plt.xlabel("Value")
            plt.ylabel("Number of Occurrences")
            plt.legend()
            plt.savefig("plots/A3C_cumsum_rewards.png")

            # Clear the figure
            plt.clf()

            # Plot histogram of rewards
            plt.hist(rewards, color="green", label="Rewards", bins=15)
            plt.title("Number of Occurrences of Rewards [A3C]")
            plt.xlabel("Value")
            plt.ylabel("Number of Occurrences")
            plt.legend()
            plt.savefig("plots/A3C_rewards.png")
        

explore_row_log('results/RowLogs/MsPacman/TB_ROW-LOG.pkl')
plt.clf() # Clear the figures in-between datasets
explore_row_log('results/RowLogs/MsPacman/A3C_ROW-LOG.pkl', TB=False)