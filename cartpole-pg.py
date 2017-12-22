import numpy as np
from collections import deque
import gym
import time
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.initializers import VarianceScaling
from keras.optimizers import Adam


def build_model():
    model = Sequential([
        Dense(4, input_shape=(4,), activation="elu", kernel_initializer=VarianceScaling()),
        Dense(2, activation="softmax")
    ])
    model.compile(optimizer=Adam(lr=0.001),
                  loss="categorical_crossentropy")
    return model


def discount_rewards(rewards, discount_rate=0.95, normalize=True):
    discounted_rewards = np.zeros_like(rewards)
    accumulator = 0
    for step in reversed(range(len(rewards))):
        accumulator = rewards[step] + accumulator * discount_rate
        discounted_rewards[step] = accumulator
    if not normalize:
        return discounted_rewards
    discounted_rewards -= np.mean(discounted_rewards)
    return discounted_rewards / np.std(discounted_rewards)


def normalize_rewards(rewards):
    rewards -= np.mean(rewards)
    return rewards / np.std(rewards)


def fit_model(model, batch):
    X = np.array([m[0] for b in batch for m in b])
    actions = np.array([m[1][0] for b in batch for m in b])
    discounted_rewards = np.concatenate([discount_rewards([m[2] for m in b], normalize=False) for b in batch])
    normalized_rewards = normalize_rewards(discounted_rewards).reshape(-1, 1)
    y = actions * normalized_rewards
    shuffle_index = list(range(len(y)))
    np.random.shuffle(shuffle_index)
    model.fit(X[shuffle_index], y[shuffle_index], epochs=1, verbose=0)
    return model


def get_action(x):
    return np.argmax(np.random.multinomial(1, x))


def train(model):
    env = gym.make("CartPole-v1")
    solved = 0
    batch = deque()
    results = deque(maxlen=100)

    for episode in range(5000):
        s = env.reset()
        episode_memory = deque()

        for step in range(500):
            a = get_action(model.predict(s.reshape(-1, 4)).flatten())
            a_1hot = np.zeros((1, 2))
            a_1hot[0, a] = 1
            s1, r, d, _ = env.step(a)
            episode_memory.append((s, a_1hot, r))
            env.render()
            s = s1

            if d:
                results.append(step)
                batch.append(episode_memory)
                if len(batch) >= 10:
                    model = fit_model(model, batch)
                    batch = deque()
                if step > 490:
                    solved += 1
                if solved > 9:
                    print("done in {} episodes.".format(episode))
                    return model
                if episode % 100 == 0:
                    print(np.mean(results), solved)
                break


if __name__ == '__main__':
    model = build_model()
    train(model)
