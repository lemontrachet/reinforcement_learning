import numpy as np
import gym
import keras
from keras.initializers import VarianceScaling
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.optimizers import Adam, RMSprop
import matplotlib.pyplot as plt
from collections import deque
import itertools


def build_model(n_inputs, n_outputs, n_hidden=48, lr=0.001):
    model = Sequential([
        Dense(256, input_shape=(n_inputs,), activation="relu", kernel_initializer='he_uniform'),
        Dense(256, activation="relu", kernel_initializer='he_uniform'),
        Dense(n_outputs, activation="linear")
    ])

    model.compile(optimizer=RMSprop(lr=lr, clipvalue=1.0),
                  loss="mse")
    return model


def transfer_weights(model, learner):
    model.set_weights(learner.get_weights())
    return model


def report_results(results, episode, episodes, epsilon, rolling):
    best = np.max([r[0] for r in results]) if results else -1
    last = results[-1][1] if results else -1
    rolling = rolling if rolling else [-1]
    roll = np.mean(rolling)
    print("episode: {}/{} - last episode: {} - global max: {} - rolling average: {} - epsilon: {}".format(
        episode, episodes, round(last, 3), round(best, 3), round(roll, 3), round(epsilon, 3)))


def fit_learner(model, replay_buffer, gamma=0.95):
    batchsize = 16
    if len(replay_buffer) < batchsize:
        return model
    memory = np.array(replay_buffer)[np.random.choice(len(replay_buffer), batchsize, replace=False), :]
    X, y = np.zeros((len(memory), 2)), np.zeros((len(memory), 2))

    for i, m in enumerate(memory):
        state, action, reward, done, next_state = m[:2], int(m[2]), m[3], m[4], m[5:]
        target = model.predict(state.reshape(-1, 2)).reshape(2)

        if done:
            total_reward = reward
        else:
            total_reward = reward + gamma * np.amax(model.predict(next_state.reshape(-1, 2)))
        target[action] = total_reward

        X[i] = state.reshape(-1, 2)
        y[i] = target.reshape(-1, 2)

    model.fit(X, y, epochs=1, verbose=0)

    return model


def play(model, learner, episodes=500, steps=200, epsilon=1.0, gamma=0.9, render=False):
    env = gym.make("MountainCar-v0")
    results = []
    highest = -1
    rolling = deque(maxlen=8)
    replay_buffer = deque(maxlen=10000)
    for episode in range(episodes):
        state = env.reset()
        if episode % 1 == 0:
            report_results(results, episode, episodes, epsilon, rolling)
        episode_memory = np.zeros((steps, 7))
        episode_max = -1

        for step in range(steps):
            env.render() if render else None
            if step % 1 == 0 and np.random.random() < epsilon:
                action = np.random.randint(2)
            elif step % 1 == 0:
                action = np.argmax(model.predict(state.reshape(-1, 2)).flatten())
            next_state, reward, done, _ = env.step(action * 2)  # 0 for 0, 2 for 1

            episode_memory[step, :2] = state
            episode_memory[step, 2] = action
            episode_memory[step, 3] = reward
            episode_memory[step, 4] = done
            episode_memory[step, 5:] = next_state

            replay_buffer.append(episode_memory[step])
            state = next_state

            episode_max = np.max([episode_max, state[0]])
            if state[0] > highest:
                highest = state[0]

            learner = fit_learner(learner, replay_buffer, gamma)

            if done:
                break

        rolling.append(episode_max)
        results.append((highest, episode_max))

        model = transfer_weights(model, learner)
        epsilon = np.max([epsilon - 0.0009, 0.05])

        if episode_max > 0.53:
            return model, learner, results

    return model, learner, results


if __name__ == '__main__':
    model = build_model(2, 2)
    learner = build_model(2, 2)
    model, learner, results = play(model, learner, episodes=2000, epsilon=0.97, render=False)
