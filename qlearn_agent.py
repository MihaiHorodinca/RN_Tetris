from keras.layers import Dense, Activation
from keras.models import Sequential, load_model
from keras.optimizers import adam_v2
import numpy as np


class ReplayBuffer(object):
    def __init__(self, buffer_size, input_shape, no_actions):
        self.buffer_i = 0
        self.buffer_size = buffer_size
        self.input_shape = input_shape
        self.no_actions = no_actions
        self.state_buffer = np.zeros((buffer_size, input_shape))
        self.next_state_buffer = np.zeros((buffer_size, input_shape))
        self.action_buffer = np.zeros((buffer_size, no_actions))
        self.reward_buffer = np.zeros(self.buffer_size)
        self.done_buffer = np.zeros(self.buffer_size)

    def store_all(self, state1, action, reward, state2, done):
        i = self.buffer_i % self.buffer_size
        self.state_buffer[i] = state1
        self.next_state_buffer[i] = state2
        self.action_buffer[i] = np.zeros(self.no_actions)
        self.action_buffer[i][action] = 1
        self.reward_buffer[i] = reward
        self.done_buffer[i] = not done

        self.buffer_i += 1

    def get_batch(self, batch_size):
        max_buffer_size = min(self.buffer_size, self.buffer_i)
        batch = np.random.choice(max_buffer_size, batch_size)

        states1 = self.state_buffer[batch]
        states2 = self.next_state_buffer[batch]
        actions = self.action_buffer[batch]
        rewards = self.reward_buffer[batch]
        dones = self.done_buffer[batch]

        return states1, actions, rewards, states2, dones


def new_Q_NN(lr, no_actions, in_d, l1_d, l2_d):
    NN = Sequential([
        Dense(l1_d, input_shape=(in_d,)),
        Activation('relu'),
        Dense(l2_d),
        Activation('relu'),
        Dense(no_actions)
    ])

    NN.compile(optimizer='adam', loss='mse')
    return NN


class Agent(object):
    def __init__(self, alpha, gamma, eps, eps_dot, eps_min, no_actions,
                 batch_size, in_d, buffer_size=100000, fname='myNN.h5'):
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.eps_dot = eps_dot
        self.eps_min = eps_min
        self.no_actions = no_actions
        self.batch_size = batch_size
        self.in_d = in_d
        self.buffer_size = buffer_size
        self.fname = fname

        self.possible_actions = [_ for _ in range(no_actions)]

        self.buffer = ReplayBuffer(self.buffer_size, self.in_d, self.no_actions)

        self.q_eval = new_Q_NN(self.alpha, self.no_actions, self.in_d, 256, 256)

    def save_all(self, state1, action, reward, state2, done):
        self.buffer.store_all(state1, action, reward, state2, done)

    def choose_action(self, state):
        state = state[np.newaxis, :]
        rand = np.random.random(1)
        if rand < self.eps:
            action = np.random.choice(self.possible_actions)
        else:
            actions = self.q_eval.predict(state)
            action = np.argmax(actions)

        return action

    def learn(self):
        if self.buffer.buffer_i < self.batch_size:
            return

        state1, action, reward, state2, done = self.buffer.get_batch(self.batch_size)

        action_val = np.array(self.possible_actions, dtype=np.int8)
        action_i = np.dot(action, action_val)
        action_i = np.array(action_i, dtype=np.int8)

        q_eval1 = self.q_eval.predict(state1)
        q_eval2 = self.q_eval.predict(state2)

        q_target = q_eval1.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        q_target[batch_index, action_i] = reward + self.gamma * np.max(q_eval2, axis=1) * done

        _ = self.q_eval.fit(state1, q_target, verbose=0)

        if self.eps > self.eps_min:
            self.eps = self.eps * self.eps_dot
        else:
            self.eps = self.eps_min

    def save_NN(self):
        self.q_eval.save(self.fname)

    def load_NN(self):
        self.q_eval = load_model(self.fname)
