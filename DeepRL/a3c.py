#!/usr/bin/env python3
"""Asynchronous Advantage Actor-Critic (A3C).
A Fork from: https://github.com/gabrieledcjr/DeepRL
"""
import logging
import numpy as np
import os
import pathlib
import signal
import sys
import threading
import time
import pandas as pd
import math
import tensorflow as tf
import csv

from threading import Event, Thread
from common_worker import CommonWorker
from a3c_training_thread import A3CTrainingThread
from common.game_state import GameState
from common.util import prepare_dir
from common.util import transform_h
from common.util import transform_h_inv
from game_ac_network import GameACFFNetwork
from queue import Queue
from copy import deepcopy
from setup_functions import *

logger = logging.getLogger("a3c")

try:
    import cPickle as pickle
except ImportError:
    import pickle


def run_a3c(args):
    """Run A3C experiment."""
    GYM_ENV_NAME = args.gym_env.replace('-', '_')
    GAME_NAME = args.gym_env.replace('NoFrameskip-v4','')

    # setup folder name and path to folder
    folder = pathlib.Path(setup_folder(args, GYM_ENV_NAME))

    # setup GPU options
    # (NOTE: no gpu support in aeolus)
    import tensorflow as tf
    gpu_options = None

    ######################################################
    # setup default device: cpu
    device = "/cpu:0"

    global_t = 0
    rewards = {'train': {}, 'eval': {}}
    best_model_reward = -(sys.maxsize)

    stop_req = False

    game_state = GameState(env_id=args.gym_env)
    action_size = game_state.env.action_space.n
    game_state.close()
    del game_state.env
    del game_state

    input_shape = (args.input_shape, args.input_shape, 4)

    #######################################################
    # setup global A3C
    GameACFFNetwork.use_mnih_2015 = args.use_mnih_2015
    global_network = GameACFFNetwork(
        action_size, -1, device, padding=args.padding,
        in_shape=input_shape)
    logger.info('A3C Initial Learning Rate={}'.format(args.initial_learn_rate))

    time.sleep(2.0)

    ############## Setup Thread Workers BEGIN ################
    # 16 total number of local threads for all experiments
    assert args.parallel_size ==16

    startIndex = 0
    all_workers = []

    # a3c learning rate and optimizer
    learning_rate_input = tf.placeholder(tf.float32, shape=(), name="opt_lr")
    grad_applier = tf.train.RMSPropOptimizer(
        learning_rate=learning_rate_input,
        decay=args.rmsp_alpha,
        epsilon=args.rmsp_epsilon)

    # setup common workers
    setup_common_worker(CommonWorker, args, action_size)

    # setup a3c workers
    setup_a3c_worker(A3CTrainingThread, args, startIndex)

    for i in range(startIndex, args.parallel_size):
        local_network = GameACFFNetwork(
            action_size, i, device=device,
            padding=args.padding,
            in_shape=input_shape)

        a3c_worker = A3CTrainingThread(
            i, global_network, local_network,
            args.initial_learn_rate, learning_rate_input, grad_applier,
            device=device, no_op_max=30)

        all_workers.append(a3c_worker)
    ############## Setup Thread Workers END ################

    # setup config for tensorflow
    config = tf.ConfigProto(
        gpu_options=gpu_options,
        log_device_placement=False,
        allow_soft_placement=True)

    # prepare sessions
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    # summary writer for tensorboard
    summ_file = args.save_to+'log/a3c/{}/'.format(GYM_ENV_NAME) + str(folder)[12:]
    summary_writer = tf.summary.FileWriter(summ_file, sess.graph)

    # init or load checkpoint with saver
    root_saver = tf.train.Saver(max_to_keep=1)
    saver = tf.train.Saver(max_to_keep=3)
    best_saver = tf.train.Saver(max_to_keep=1)

    checkpoint = tf.train.get_checkpoint_state(str(folder)+'/model_checkpoints')
    if checkpoint and checkpoint.model_checkpoint_path:
        root_saver.restore(sess, checkpoint.model_checkpoint_path)
        logger.info("checkpoint loaded:{}".format(
            checkpoint.model_checkpoint_path))
        tokens = checkpoint.model_checkpoint_path.split("-")

        # restore global step
        global_t = int(tokens[-1])
        logger.info(">>> global step set: {}".format(global_t))

        # restore wall time
        wall_t_fname = folder / 'wall_t'
        with wall_t_fname.open('r') as f:
            wall_t = float(f.read())

        # restore reward files
        best_reward_file = folder / 'model_best/best_model_reward'
        with best_reward_file.open('r') as f:
            best_model_reward = float(f.read())
        reward_file = folder / '{}-a3c-rewards.pkl'.format(GYM_ENV_NAME)
        rewards = pickle.load(reward_file.open('rb'))
    else:
        logger.warning("Could not find old checkpoint")
        # set wall time
        wall_t = 0.0
        prepare_dir(folder, empty=True)
        prepare_dir(folder / 'model_checkpoints', empty=True)
        prepare_dir(folder / 'model_best', empty=True)
        prepare_dir(folder / 'frames', empty=True)

    lock = threading.Lock()

    def next_t(current_t, freq):
        return np.ceil((current_t + 0.00001) / freq) * freq

    next_global_t = next_t(global_t, args.eval_freq)
    next_save_t = next_t(
        global_t, args.eval_freq*args.checkpoint_freq)

    step_t = 0

    last_temp_global_t = global_t

    def train_function(parallel_idx, th_ctr):
        nonlocal global_t, step_t, rewards, lock, next_global_t, \
            next_save_t, last_temp_global_t

        parallel_worker = all_workers[parallel_idx]
        parallel_worker.set_summary_writer(summary_writer)

        with lock:
            # Evaluate model before training
            if not stop_req and global_t == 0 and step_t == 0:
                # Log the reward, and the transformed reward
                reward_t = parallel_worker.testing(sess, args.eval_max_steps, global_t, folder, worker=all_workers[-1])[0]
                trans_reward_t = transform_h(reward_t + 0.99 * transform_h_inv(reward_t))
                rewards['eval'][step_t] = [reward_t, trans_reward_t]
                # save checkpoint
                checkpt_file = folder / 'model_checkpoints'
                checkpt_file /= '{}_checkpoint'.format(GYM_ENV_NAME)
                saver.save(sess, str(checkpt_file), global_step=global_t)
                save_best_model(rewards['eval'][global_t][0])
                # dump reward pickle
                reward_fname = folder / '{}-a3c-rewards.pkl'.format(GYM_ENV_NAME)
                pickle.dump(rewards, reward_fname.open('wb'), pickle.HIGHEST_PROTOCOL)
                logger.info('Dump pickle at step {}'.format(global_t))

                step_t = 1

        # set start_time
        start_time = time.time() - wall_t
        parallel_worker.set_start_time(start_time)

        while True:
            if stop_req:
                return

            if global_t >= (args.max_time_step * args.max_time_step_fraction):
                return

            # a3c training thread worker
            th_ctr.get()

            train_out = parallel_worker.train(sess, global_t, rewards)
            diff_global_t, episode_end, part_end = train_out

            th_ctr.put(1)

            # ensure only one thread is updating global_t at a time
            with lock:
                global_t += diff_global_t
                # if during a thread's update, global_t has reached a evaluation interval
                if global_t > next_global_t:
                    next_global_t = next_t(global_t, args.eval_freq)
                    step_t = int(next_global_t - args.eval_freq)
                    # wait 30min max for all threads to finish before testing,
                    # in case one thread fails
                    time_waited = 0
                    while not stop_req and th_ctr.qsize() < len(all_workers):
                        time.sleep(0.01)
                        time_waited += 0.01
                        if time_waited > 1800:
                            sys.exit('Exceed waiting time, exiting...')

                    step_t = int(next_global_t - args.eval_freq)
                    # test A3C agent
                    reward_t = parallel_worker.testing(sess, args.eval_max_steps, global_t, folder, worker=all_workers[-1])[0]
                    trans_reward_t = transform_h(reward_t + 0.99 * transform_h_inv(reward_t))
                    rewards['eval'][step_t] = [reward_t, trans_reward_t]
                    #print("\n\n\nGLOBAL_T: {}\nREWARD: {}\nTRANS REWARD: {}\ R[EVAL]: {}\n\n\n".format(global_t, reward_t, trans_reward_t, rewards['eval'][step_t]))
                    save_best_model(rewards['eval'][step_t][0])

                if global_t > next_save_t:
                    next_save_t = next_t(global_t, args.eval_freq*args.checkpoint_freq)
                    # save a3c
                    checkpt_file = folder / 'model_checkpoints'
                    checkpt_file /= '{}_checkpoint'.format(GYM_ENV_NAME)
                    saver.save(sess, str(checkpt_file), global_step=global_t,
                            write_meta_graph=False)
                    # dump pickle
                    reward_fname = folder / '{}-a3c-rewards.pkl'.format(GYM_ENV_NAME)
                    pickle.dump(rewards, reward_fname.open('wb'), pickle.HIGHEST_PROTOCOL)
                    logger.info('Dump pickle at step {}'.format(global_t))

    def signal_handler(signal, frame):
        nonlocal stop_req
        logger.info('You pressed Ctrl+C!')
        stop_req = True
        if stop_req and global_t == 0:
            sys.exit(1)


    def save_best_model(test_reward):
        nonlocal best_model_reward
        if test_reward > best_model_reward:
            best_model_reward = test_reward
            best_reward_file = folder / 'model_best/best_model_reward'

            with best_reward_file.open('w') as f:
                f.write(str(best_model_reward))

            best_checkpt_file = folder / 'model_best'
            best_checkpt_file /= '{}_checkpoint'.format(GYM_ENV_NAME)
            best_saver.save(sess, str(best_checkpt_file))


    train_threads = []
    th_ctr = Queue()
    for i in range(args.parallel_size):
        th_ctr.put(1)

    for i in range(args.parallel_size):
        worker_thread = Thread(
            target=train_function,
            args=(i, th_ctr, ))
        train_threads.append(worker_thread)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # set start time and start threads
    start_time = time.time() - wall_t
    for t in train_threads:
        t.start()

    print('Press Ctrl+C to stop')

    # end of training
    for t in train_threads:
        t.join()

    logger.info('Now saving data. Please wait')

    # write wall time
    wall_t = time.time() - start_time
    wall_t_fname = folder / 'wall_t'
    with wall_t_fname.open('w') as f:
        f.write(str(wall_t))

    checkpoint_file = str(folder / '{}_checkpoint_a3c'.format(GYM_ENV_NAME))
    root_saver.save(sess, checkpoint_file, global_step=global_t)

    reward_fname = folder / '{}-a3c-rewards.pkl'.format(GYM_ENV_NAME)
    pickle.dump(rewards, reward_fname.open('wb'), pickle.HIGHEST_PROTOCOL)
    logger.info('Data saved!')

    sess.close()
