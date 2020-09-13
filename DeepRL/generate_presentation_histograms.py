import numpy as np
import pickle
import matplotlib.pyplot as plt

# Logging class files (A3C or TB Row Logs)
a3c_row_logs_pkl = "results/TRIAL1-MsPacman/A3C_ROW-LOG.pkl"
tb_row_logs_pkl = "results/TRIAL1-MsPacman/TB_ROW-LOG.pkl"

# "Raw" dumped output
# TB 
tb_trans_returns_pkl = "/Users/nburley/A3C_TB_Research/results/TB_RewardLogs/tb_transformed_returns.pkl"
tb_raw_returns_pkl = "/Users/nburley/A3C_TB_Research/results/TB_RewardLogs/raw_returns.pkl"
tb_raw_rewards_pkl = "/Users/nburley/A3C_TB_Research/results/TB_RewardLogs/tp_raw_rewards.pkl" # Need to rename this one, and delete the OG
# A3C
a3c_clip_returns_pkl = "/Users/nburley/A3C_TB_Research/results/A3C_RewardLogs/a3c_clipped_returns.pkl"
a3c_clip_rewards_pkl = "/Users/nburley/A3C_TB_Research/results/A3C_RewardLogs/a3c_clipped_rewards.pkl"


# Function that handles the nested array output (changes [[1,2,3], [4,5,6]] to [1,2,3,4,5,6])
def noSubArr(og_arr):
    no_sub_arrs = []
    for sub_arr in og_arr:
        for val in sub_arr:
            no_sub_arrs.append(val)
    return no_sub_arrs


# Generate task histograms from the "raw" dumped arrays
def generateResultHist(tb_trans_returns_pkl, tb_raw_returns_pkl, tb_raw_rewards_pkl, a3c_clip_returns_pkl, a3c_clip_rewards_pkl):
    with open(tb_trans_returns_pkl, 'rb') as tb_trans_file, open(tb_raw_returns_pkl, 'rb') as tb_raw_ret_file, open(tb_raw_rewards_pkl, 'rb') as tb_raw_rew_file, open(a3c_clip_returns_pkl, 'rb') as a3c_ret_file, open(a3c_clip_rewards_pkl, 'rb') as a3c_rew_file:

        # Read in the TB Transformed Returns
        tb_trans_returns = []
        while True:
            try:
                batch = pickle.load(tb_trans_file)
                for val in batch:
                    tb_trans_returns.append(val)
            except EOFError:
                break
        tb_trans_file.close()
        
        # Read in the TB Raw Returns
        tb_raw_returns = []
        while True:
            try:
                batch = pickle.load(tb_raw_ret_file)
                for val in batch:
                    tb_raw_returns.append(val)
            except EOFError:
                break
        tb_raw_ret_file.close()
        
        # Read in the TB Raw Rewards
        tb_raw_rewards = []
        while True:
            try:
                batch = pickle.load(tb_raw_rew_file)
                for val in batch:
                    tb_raw_rewards.append(val)
            except EOFError:
                break
        tb_raw_rew_file.close()
        
        # Read in the A3C Clipped Returns
        a3c_clip_returns = []
        while True:
            try:
                batch = pickle.load(a3c_ret_file)
                for val in batch:
                    a3c_clip_returns.append(val)
            except EOFError:
                break
        a3c_ret_file.close()
        
        # Read in the A3C Clipped Rewards
        a3c_clip_rewards = []
        while True:
            try:
                batch = pickle.load(a3c_rew_file)
                for val in batch:
                    a3c_clip_rewards.append(val)
            except EOFError:
                break
        a3c_rew_file.close()

        # Generate the RETURN histogram
        plt.hist(tb_trans_returns, color="green", label="TB Transformed Returns", alpha=0.5, bins=15)
        plt.hist(tb_raw_returns, color="blue", label="TB Raw Returns", alpha=0.5, bins=15)
        plt.hist(a3c_clip_returns, color="yellow", label="A3C Clipped Returns", alpha=0.5, bins=15)
        plt.title("Histogram of TB Transformed Returns, TB Raw Returns, and A3C Clipped Returns")
        plt.xlabel("Value")
        plt.ylabel("Number of Occurrences")
        plt.legend()
        plt.savefig("plots/assignment_return_plot.png")
        
        # Clear the plot
        plt.clf()
        
        # Generate the REWARD histogram
        plt.hist(tb_raw_rewards, color="green", label="TB Raw Rewards", alpha=0.5, bins=15)
        plt.hist(a3c_clip_rewards, color="yellow", label="A3C Rewards", alpha=0.5, bins=15)
        plt.title("Histogram of A3C Clipped vs. TB Raw Rewards")
        plt.xlabel("Value")
        plt.ylabel("Number of Occurrences")
        plt.legend()
        plt.savefig("plots/assignment_reward_plot.png")


# Actually generate the plots!
generateResultHist(tb_trans_returns_pkl, tb_raw_returns_pkl, tb_raw_rewards_pkl, a3c_clip_returns_pkl, a3c_clip_rewards_pkl)

"""
# Generate task histograms from the Row Log logging method (task requirement C, UGproject_taskSept4.pdf)
def generate_RowLog_ResultHist(a3c_row_logs_pkl, tb_row_logs_pkl):
    with open(a3c_row_logs_pkl, 'rb') as a3c_file, open(tb_row_logs_pkl, 'rb') as tb_file:
        # Read in the A3C returns
        a3c_returns = []
        a3c_rewards = []
        while True:
            try:
                batch = pickle.load(a3c_file)
                for reward, batch_return in zip(batch.rewards, batch.batch_cumsum_rewards):
                    a3c_returns.append(batch_return)
                    a3c_rewards.append(reward)
            except:
                break
        
        # Close the file
        a3c_file.close()
        
        # Read in the TB transformed returns and raw returns
        raw_returns = []
        tb_returns = []
        tb_rewards = []
        while True:
            try:
                batch = pickle.load(tb_file)
                for reward, raw_return, batch_return in zip(batch.rewards, batch.batch_raw_rewards, batch.batch_cumsum_rewards):
                    raw_returns.append(raw_return)
                    tb_returns.append(batch_return)
                    tb_rewards.append(reward)
            except:
                break
        
        # Close the other file
        tb_file.close()
        
        # Actually generate the RETURN histogram
        # If, like last time, you can't see the TB results, that means it didn't train enough
        
        plt.hist(raw_returns, color="blue", label="TB Raw Returns", alpha=0.5, bins=15)
        plt.hist(tb_returns, color="green", label="TB Transformed Returns", alpha=0.5, bins=15)
        plt.hist(a3c_returns, color="yellow", label="A3C Clipped Returns", alpha=0.5, bins=15)
        plt.title("Histogram of A3C Clipped Returns")
        plt.xlabel("Value")
        plt.ylabel("Number of Occurrences")
        plt.legend()
        plt.savefig("plots/assignment_return_plot.png")
        
        # Clear the plot
        plt.clf()
        
        # Generate the REWARD histogram
        plt.hist(tb_rewards, color="green", label="TB Rewards", alpha=0.5, bins=15)
        plt.hist(a3c_rewards, color="yellow", label="A3C Rewards", alpha=0.5, bins=15)
        plt.title("Histogram of A3C vs. TB Rewards")
        plt.xlabel("Value")
        plt.ylabel("Number of Occurrences")
        plt.legend()
        plt.savefig("plots/assignment_reward_plot.png")

generate_RowLog_ResultHist(a3c_row_logs_pkl, tb_row_logs_pkl)
"""