import os
import logging
import numpy as np

from common.util import load_memory

logger = logging.getLogger("a3c")

def setup_folder(args, env_name):
    if not os.path.exists(args.save_to+"a3c/"):
        os.makedirs(args.save_to+"a3c/")

    if args.folder is not None:
        folder = args.save_to+'a3c/{}_{}'.format(env_name, args.folder)
    else:
        folder = args.save_to+'a3c/{}'.format(env_name)
        end_str = ''

        if args.unclipped_reward:
            end_str += '_rawreward'

        if args.transformed_bellman:
            end_str += '_TB'

        folder += end_str

    if args.append_experiment_num is not None:
        folder += '_' + args.append_experiment_num

    return folder


def setup_common_worker(CommonWorker, args, action_size):
    CommonWorker.action_size = action_size
    CommonWorker.env_id = args.gym_env
    CommonWorker.reward_constant = args.reward_constant
    CommonWorker.max_global_time_step = args.max_time_step
    if args.unclipped_reward:
        CommonWorker.reward_type = "RAW"
    else:
        CommonWorker.reward_type = "CLIP"


def setup_a3c_worker(A3CTrainingThread, args, log_idx):
    A3CTrainingThread.log_interval = args.log_interval
    A3CTrainingThread.perf_log_interval = args.performance_log_interval
    A3CTrainingThread.local_t_max = args.local_t_max
    A3CTrainingThread.entropy_beta = args.entropy_beta
    A3CTrainingThread.gamma = args.gamma
    A3CTrainingThread.use_mnih_2015 = args.use_mnih_2015
    A3CTrainingThread.transformed_bellman = args.transformed_bellman
    A3CTrainingThread.clip_norm = args.grad_norm_clip
    A3CTrainingThread.log_idx = log_idx
    A3CTrainingThread.reward_constant = args.reward_constant
