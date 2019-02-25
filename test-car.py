#!/usr/bin/env python

import time
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.optimizers import Adam
from PIL import Image

def decode_action(action):
  '''
  decision = [0, 0, 0]
  if action < 36 :
    decision[0] = (18 - action) / 18
  elif action < 46:
    decision[1] = (action - 36) / 10
  elif action < 56:
    decision[2] = (action - 46) / 10
  '''
  decision = [0, 0.3, 0]
  decision[0] = (action - 18) / 36
  return decision

steps = 1000

class DQNAgent:
  def __init__(self, state_size, action_size):
    self.img_w = 32
    self.img_h = 32
    self.state_size = state_size
    self.action_size = action_size
    self.memory = deque(maxlen=2000)
    self.gamma = 0.95    # discount rate
    self.epsilon = 1.0  # exploration rate
    self.epsilon_min = 0.01
    self.epsilon_decay = 0.995
    self.learning_rate = 0.001
    self.model = self._build_model()

  def _build_model(self):
    # Neural Net for Deep-Q learning Model
    model = Sequential()
    model.add(Conv2D(12, kernel_size=3, input_shape=(self.img_h, self.img_h, 1), activation='relu'))
    model.add(Conv2D(12, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(24, activation='relu'))
    model.add(Dense(self.action_size, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
    model.summary()
    return model

  def remember(self, state, action, reward, next_state, done):
    self.memory.append((state, action, reward, next_state, done))

  def act(self, state):
    if np.random.rand() <= self.epsilon:
      return random.randrange(self.action_size)
    input_img = self.preprocess_state(state)
    act_values = self.model.predict(input_img)
    return np.argmax(act_values[0])  # returns action

  def replay(self, batch_size):
    minibatch = random.sample(self.memory, batch_size)
    for state, action, reward, next_state, done in minibatch:
      target = reward
      if not done:
        input_img = self.preprocess_state(next_state)
        target = (reward + self.gamma * np.amax(self.model.predict(input_img)[0]))
      input_img = self.preprocess_state(state)
      target_f = self.model.predict(input_img)
      target_f[0][action] = target
      self.model.fit(input_img, target_f, epochs=1, verbose=0)
    if self.epsilon > self.epsilon_min:
      self.epsilon *= self.epsilon_decay

  def preprocess_state(self, state):
    img = Image.fromarray(np.uint8(state[0]))
    gray = img.convert('L')
    resized = gray.resize((self.img_w, self.img_h), Image.ANTIALIAS)
    output = np.array(resized)
    output = np.reshape(output, (1, self.img_w, self.img_h, 1))
    return output

  def load(self, name):
    self.model.load_weights(name)

  def save(self, name):
    self.model.save_weights(name, overwrite=True)

if __name__ == "__main__":
  env = gym.make('CarRacing-v0')
  state_size = env.observation_space.shape
  #action_size = env.action_space.shape[0]
  action_size = steering_actions = 36
  agent = DQNAgent(state_size, action_size)
  agent.load("car-dqn.h5")
  done = False
  batch_size = 32
  state = env.reset()
  state = np.reshape(state, (1,) + state_size)

  for e in range(steps):
    env.render()
    action = agent.act(state)
    decision = decode_action(action)
    state, reward, done, _ = env.step(decision)
    state = np.reshape(state, (1,) + state_size)
