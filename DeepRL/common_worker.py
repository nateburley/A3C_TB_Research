import cv2
import logging
import tensorflow as tf
import numpy as np

from common.util import generate_image_for_cam_video
from common.util import grad_cam
from common.util import make_movie
from common.util import visualize_cam
from common.game_state import get_wrapper_by_name
from termcolor import colored

logger = logging.getLogger("common_worker")

class CommonWorker(object):
    thread_idx = -1
    action_size = -1
    reward_type = 'CLIP' # CLIP | RAW
    env_id = None
    reward_constant = 0
    max_global_time_step=0

    def pick_action(self, logits):
        """Choose action probabilistically.

        Reference:
        https://github.com/ppyht2/tf-a2c/blob/master/src/policy.py
        """
        noise = np.random.uniform(0, 1, np.shape(logits))
        return np.argmax(logits - np.log(-np.log(noise)))


    def set_start_time(self, start_time):
        """Set start time."""
        self.start_time = start_time


    def set_summary_writer(self, writer):
        """Set summary writer."""
        self.writer = writer


    def _anneal_learning_rate(self, global_time_step, initial_learning_rate):
        learning_rate = initial_learning_rate * \
        (self.max_global_time_step - global_time_step) / \
        self.max_global_time_step
        if learning_rate < 0.0:
            learning_rate = 0.0
        return learning_rate


    def record_summary(self, score=0, steps=0, episodes=None, global_t=0,
                       mode='Test'):
        """Record summary."""
        summary = tf.Summary()
        summary.value.add(tag='{}/score'.format(mode),
                          simple_value=float(score))
        summary.value.add(tag='{}/steps'.format(mode),
                          simple_value=float(steps))
        if episodes is not None:
            summary.value.add(tag='{}/episodes'.format(mode),
                              simple_value=float(episodes))
        self.writer.add_summary(summary, global_t)
        self.writer.flush()


    def testing(self, sess, max_steps, global_t, folder, worker=None):
        """Evaluate A3C."""
        assert worker is not None

        logger.info("Evaluate policy at global_t={}...".format(global_t))

        # copy weights from global to local
        sess.run(worker.sync)

        episode_buffer = []
        worker.game_state.reset(hard_reset=True)
        episode_buffer.append(worker.game_state.get_screen_rgb())

        total_reward = 0
        total_steps = 0
        episode_reward = 0
        episode_steps = 0
        n_episodes = 0
        while max_steps > 0:
            state = cv2.resize(worker.game_state.s_t,
                               worker.local_net.in_shape[:-1],
                               interpolation=cv2.INTER_AREA)
            pi_, value_, logits_ = worker.local_net.run_policy_and_value(
                sess, state)

            if False:
                action = np.random.choice(range(worker.action_size), p=pi_)
            else:
                action = worker.pick_action(logits_)

            # take action
            worker.game_state.step(action)
            terminal = worker.game_state.terminal

            if n_episodes == 0 and global_t % 5000000 == 0:
                episode_buffer.append(worker.game_state.get_screen_rgb())

            episode_reward += worker.game_state.reward
            episode_steps += 1
            max_steps -= 1

            # s_t = s_t1
            worker.game_state.update()

            if terminal:
                env = worker.game_state.env
                name = 'EpisodicLifeEnv'
                if get_wrapper_by_name(env, name).was_real_done:
                    if n_episodes == 0 and global_t % 5000000 == 0:
                        time_per_step = 0.0167
                        images = np.array(episode_buffer)
                        file = 'frames/image{ep:010d}'.format(ep=global_t)
                        duration = len(images)*time_per_step
                        make_movie(images, str(folder / file),
                                   duration=duration, true_image=True,
                                   salience=False)
                        episode_buffer = []
                    n_episodes += 1
                    score_str = colored("score={}".format(episode_reward),
                                        "yellow")
                    steps_str = colored("steps={}".format(episode_steps),
                                        "cyan")
                    log_data = (global_t, worker.thread_idx, self.thread_idx,
                                n_episodes, score_str, steps_str,
                                total_steps)
                    logger.debug("test: global_t={} test_worker={} cur_worker={}"
                                 " trial={} {} {}"
                                 " total_steps={}".format(*log_data))
                    total_reward += episode_reward
                    total_steps += episode_steps
                    episode_reward = 0
                    episode_steps = 0

                worker.game_state.reset(hard_reset=False)

        if n_episodes == 0:
            total_reward = episode_reward
            total_steps = episode_steps
        else:
            total_reward = total_reward / n_episodes
            total_steps = total_steps // n_episodes

        log_data = (global_t, worker.thread_idx, self.thread_idx,
                    total_reward, total_steps,
                    n_episodes)
        logger.info("test: global_t={} test_worker={} cur_worker={}"
                    " final score={} final steps={}"
                    " # trials={}".format(*log_data))

        worker.record_summary(
            score=total_reward, steps=total_steps,
            episodes=n_episodes, global_t=global_t, mode='A3C_Test')

        # reset variables used in training
        worker.episode_reward = 0
        worker.episode_steps = 0
        worker.game_state.reset(hard_reset=True)
        worker.last_rho = 0.

        return (total_reward, total_steps, n_episodes)
