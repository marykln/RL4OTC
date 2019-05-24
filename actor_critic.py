import gym
import random
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.models import Model
from keras.layers import Dense, Flatten, Input
from keras.layers.convolutional import Conv2D
from obstacle_tower_env import ObstacleTowerEnv

global episode
episode = 0
EPISODES = 8000000


class TestAgent:
    def __init__(self, action_size):
        self.state_size = (84, 84, 4)
        self.action_size = action_size

        self.discount_factor = 0.99
        self.no_op_steps = 30

        self.actor, self.critic = self.build_model()

    def build_model(self):
        input = Input(shape=self.state_size)
        conv = Conv2D(16, (8, 8), strides=(4, 4), activation='relu')(input)
        conv = Conv2D(32, (4, 4), strides=(2, 2), activation='relu')(conv)
        conv = Flatten()(conv)
        fc = Dense(256, activation='relu')(conv)
        policy = Dense(self.action_size, activation='softmax')(fc)
        value = Dense(1, activation='linear')(fc)

        actor = Model(inputs=input, outputs=policy)
        critic = Model(inputs=input, outputs=value)

        actor.summary()
        critic.summary()

        return actor, critic

    def get_action(self, history):
        history = np.float32(history / 255.)
        policy = self.actor.predict(history)[0]

        action_index = np.argmax(policy)
        return action_index

    def load_model(self, name):
        self.actor.load_weights(name)

def pre_processing(next_observe, observe):
    processed_observe = np.maximum(next_observe, observe)
    processed_observe = np.uint8(
        resize(rgb2gray(processed_observe), (84, 84), mode='constant') * 255)
    return processed_observe


if __name__ == "__main__":
    env = ObstacleTowerEnv('./ObstacleTower/obstacletower', retro=True, realtime_mode=True)
    agent = TestAgent(action_size=3)
    # agent.load_model("save_model/tower_a3c_actor.h5")

    step = 0

    while episode < EPISODES:
        done = False
        dead = False

        score, start_life = 0, 5
        #observe = env.reset()
        #next_observe = observe

        # tower 0, floor 10 = second room holds key
        config = {'tower-seed': 0, 'starting-floor': 10, 'dense-reward': 1, 'agent-perspective': 1, 'allowed-rooms': 1,
                  'allowed-modules': 0,
                  'allowed-floors': 0}
        obs = env.reset(config=config)
        next_observe = obs

        for _ in range(random.randint(1, 20)):
            observe = next_observe
            next_observe, _, _, _ = env.step(1)

        state = pre_processing(next_observe, observe)
        history = np.stack((state, state, state, state), axis=2)
        history = np.reshape([history], (1, 84, 84, 4))
        print(history)

        while not done:
            env.render()
            step += 1
            observe = next_observe

            action = agent.get_action(history)
            print("Action:" + str(action))

            fake_action = action

            next_observe, reward, done, info = env.step(fake_action)

            next_state = pre_processing(next_observe, observe)
            next_state = np.reshape([next_state], (1, 84, 84, 1))
            next_history = np.append(next_state, history[:, :, :, :3], axis=3)

            score += reward

            # if agent is dead, then reset the history
            if dead:
                history = np.stack(
                    (next_state, next_state, next_state, next_state), axis=2)
                history = np.reshape([history], (1, 84, 84, 4))
            else:
                history = next_history

            # if done, plot the score over episodes
            if done:
                episode += 1
                print("episode:", episode, "  score:", score, "  step:", step)
                step = 0

