import numpy as np
import pickle
import matplotlib.pyplot as plt

a3c_row_logs_pkl = "results/TRIAL1-MsPacman/A3C_ROW-LOG.pkl"
tb_row_logs_pkl = "results/TRIAL1-MsPacman/TB_ROW-LOG.pkl"

# Generate RETURN histogram (task requirement C, UGproject_taskSept4.pdf)
def generateResultHist(a3c_row_logs_pkl, tb_row_logs_pkl):
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

generateResultHist(a3c_row_logs_pkl, tb_row_logs_pkl)