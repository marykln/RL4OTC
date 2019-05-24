import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize
import tensorflow as tf
from obstacle_tower_env import ObstacleTowerEnv
from matplotlib import pyplot as plt
from PIL import Image


OUTPUT_GRAPH = True
MAX_EPISODE = 3000
DISPLAY_REWARD_THRESHOLD = 200  # renders environment if total episode reward is greater then this threshold
MAX_EP_STEPS = 1000 # maximum time step in one episode
RENDER = False # rendering wastes time
GAMMA = 0.9 # reward discount in TD error
LR_A = 0.001 # actor learning rate
LR_C = 0.01 # critic learning rate


class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features, n_features], "state")
        self.a = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error

        self.s = self.s[:, :, :, np.newaxis]  # artifact to create 4dim space (1st dim = batchsize)
        # print("SELF.S SHAPE: " + str(self.s.shape))
        with tf.variable_scope('Actor'):
            self.input_layer = tf.layers.conv2d(
                inputs=self.s,
                filters=2 * 2,
                kernel_size=[2, 2],
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='input_layer'
            )

            self.scnd_conv = tf.layers.conv2d(
                inputs=self.input_layer,
                filters=2 * 2,
                kernel_size=[2, 2],
                activation=tf.nn.relu,  # get action probabilities
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='scnd_conv'
            )

            self.flatten = tf.layers.flatten(
                inputs=self.scnd_conv,
                name='flatten'
            )

            self.dense = tf.layers.dense(
                inputs=self.flatten,
                units=200,
                activation=tf.nn.relu,
                name='1st_dense'
            )

            self.acts_prob = tf.layers.dense(
                inputs=self.dense,
                units=54,
                activation=tf.nn.softmax,
                name='acts_prob'
            )

        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.acts_prob[0, self.a])
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)

    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    # choose action based on previous state s
    def choose_action(self, s):
        s = s[np.newaxis, :]
        print(" S :" + str(s.shape)) # shape: (1, 84, 84, 1)
        # wants s to be of dimsize self.s (1, 84, 84, 1)
        probs = self.sess.run(self.acts_prob, {self.s: s})  # get probabilities for all actions
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())  # return a int


class Critic(object):
    def __init__(self, sess, n_features, lr=0.01):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features, n_features], "state")
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')

        # self.s = np.expand_dims(self.s, axis=3)
        self.s = self.s[:, :, :, np.newaxis]  # artifact to create 4dim space (1st dim = batchsize)
        with tf.variable_scope('Critic'):
            self.input_layer = tf.layers.conv2d(
                inputs=self.s,
                filters=2 * 2,
                kernel_size=[2, 2],
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='input_layer'
            )

            self.scnd_conv = tf.layers.conv2d(
                inputs=self.input_layer,
                filters=2 * 2,
                kernel_size=[2, 2],
                activation=tf.nn.relu,  # get action probabilities
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='scnd_conv'
            )

            self.flatten = tf.layers.flatten(
                inputs=self.scnd_conv,
                name='flatten'
            )

            self.dense = tf.layers.dense(
                inputs=self.flatten,
                units=200,
                activation=tf.nn.relu,
                name='1st_dense'
            )

            self.v = tf.layers.dense(
                inputs=self.dense,
                units=1,
                activation=tf.nn.softmax,
                name='V'
            )

        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + GAMMA * self.v_ - self.v
            self.loss = tf.square(self.td_error)  # TD_error = (r+gamma*V_next) - V_eval
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                    {self.s: s, self.v_: v_, self.r: r})
        return td_error

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

if __name__ == "__main__":
    # Set retro=True to get integers for every action instead of MultiDiscrete
    env = ObstacleTowerEnv('./ObstacleTower/obstacletower', retro=True, realtime_mode=True)
    N_F = env.observation_space.shape[0] # number features 84
    N_A = env.action_space.n # number actions 54

    tf.reset_default_graph()
    #sess = tf.Session()
    config = tf.ConfigProto()
    config.gpu_options.allocator_type = 'BFC'
    sess = tf.Session(config=config)
    writer = tf.summary.FileWriter('./graphs', sess.graph)

    actor = Actor(sess, n_features=N_F, n_actions=N_A, lr=LR_A)
    critic = Critic(sess, n_features=N_F, lr=LR_C)     # we need a good teacher, so the teacher should learn faster than the actor

    sess.run(tf.global_variables_initializer())

    if OUTPUT_GRAPH:
        tf.summary.FileWriter("logs/", sess.graph)

    for i_episode in range(MAX_EPISODE):
        s = env.reset()
        config = {'tower-seed': 0, 'starting-floor': 10, 'dense-reward': 1, 'agent-perspective': 1, 'allowed-rooms': 1,
                  'allowed-modules': 0,
                  'allowed-floors': 0}
        obs = env.reset(config=config)
        action = 18
        obs, reward, done, info = env.step(action)
        obs, reward, done, info = env.step(action)
        obs, reward, done, info = env.step(action)
        obs, reward, done, info = env.step(action)
        obs, reward, done, info = env.step(action)
        obs, reward, done, info = env.step(action)
        obs, reward, done, info = env.step(action)
        obs, reward, done, info = env.step(action)
        obs, reward, done, info = env.step(action)
        obs, reward, done, info = env.step(action)
        obs, reward, done, info = env.step(action)
        obs, reward, done, info = env.step(action)
        obs, reward, done, info = env.step(action)
        obs, reward, done, info = env.step(action)
        obs, reward, done, info = env.step(action)
        obs, reward, done, info = env.step(action)
        obs, reward, done, info = env.step(action)
        obs, reward, done, info = env.step(action)
        obs, reward, done, info = env.step(action)
        obs, reward, done, info = env.step(action)
        obs, reward, done, info = env.step(action)
        obs, reward, done, info = env.step(action)
        obs, reward, done, info = env.step(action)
        obs, reward, done, info = env.step(action)
        obs, reward, done, info = env.step(action)
        obs, reward, done, info = env.step(action)
        obs, reward, done, info = env.step(action)
        action = 6
        obs, reward, done, info = env.step(action)
        obs, reward, done, info = env.step(action)
        obs, reward, done, info = env.step(action)
        obs, reward, done, info = env.step(action)
        obs, reward, done, info = env.step(action)
        obs, reward, done, info = env.step(action)
        obs, reward, done, info = env.step(action)
        obs, reward, done, info = env.step(action)
        obs, reward, done, info = env.step(action)
        obs, reward, done, info = env.step(action)
        obs, reward, done, info = env.step(action)
        obs, reward, done, info = env.step(action)
        obs, reward, done, info = env.step(action)
        obs, reward, done, info = env.step(action)
        obs, reward, done, info = env.step(action)
        action = 18
        obs, reward, done, info = env.step(action)
        obs, reward, done, info = env.step(action)
        obs, reward, done, info = env.step(action)
        obs, reward, done, info = env.step(action)
        s, reward, done, info = env.step(action)
        s = rgb2gray(s)
        s = np.expand_dims(s, axis=2)

        t = 0
        track_r = []
        track_a = []
        track_s = []
        while True:
            env.render()

            a = actor.choose_action(s)  # state of dim (1, 84, 84) fed to actor to choose the next action based on the current state

            s_, r, done, info = env.step(a)
            s_ = rgb2gray(s_)
            s_ = np.expand_dims(s_, axis=2)

            if done:
                r = -20

            track_r.append(r)
            track_a.append(a)
            track_s.append(s)

            td_error = critic.learn(s, r, s_)  # gradient = grad[r + gamma * V(s_) - V(s)]
            actor.learn(s, a, td_error)  # true_gradient = grad[logPi(s,a) * td_error]
            print("Td_Error: " + str(td_error) + "  Reward: " + str(r) + "  Action: " + str(a))
            s = s_
            t += 1

            if done or t >= MAX_EP_STEPS:
                ep_rs_sum = sum(track_r)

                if 'running_reward' not in globals():
                    running_reward = ep_rs_sum
                else:
                    running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
                if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True  # rendering
                print("episode:", i_episode, "  reward:", int(running_reward))
                break
