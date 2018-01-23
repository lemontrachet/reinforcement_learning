import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Reshape
from keras.initializers import VarianceScaling
from keras.optimizers import RMSprop
import threading
import time
from queue import Queue
import tensorflow as tf
from collections import deque
import gym
from maze_env import Maze
from applegrid import AppleGrid


#ENV = "CartPole-v1"
#ENV = "MountainCar-v0"
STATE = 4
STEPS = 50
BEST = -1
MONITOR_Q = Queue()
THREADS = 7


def build_model():
    model = Sequential()
    model.add(Dense(64, input_shape=(4,), activation="elu", kernel_initializer=VarianceScaling()))
    model.add(Dense(64, activation="elu", kernel_initializer=VarianceScaling()))
    model.add(Dense(4, activation="linear", kernel_initializer=VarianceScaling()))
    model.compile(loss="mse", optimizer=RMSprop(lr=0.001))
    model._make_predict_function()
    return model
"""

def build_model():
    model = Sequential()
    model.add(Conv2D(16, 4, strides=4, padding="SAME", data_format="channels_last",
                     input_shape=[10, 10, 1], activation="relu",
                     kernel_initializer=VarianceScaling()))
    model.add(Conv2D(64, 8, strides=2, padding="SAME", activation="relu",
                     kernel_initializer=VarianceScaling()))
    model.add(Flatten())
    #model.add(Dense(64, activation="relu", kernel_initializer=VarianceScaling()))
    model.add(Dense(4, activation="linear", kernel_initializer=VarianceScaling()))
    model.compile(loss="mse", optimizer=RMSprop(lr=0.001))
    model._make_predict_function()
    return model
"""

def play(model, graph, q, epsilon, n):
    """
    play 5 episodes with a local model
    """
    global BEST, MONITOR_Q
    env = AppleGrid()
    with graph.as_default():
        for episode in range(5):
            rewards = 0
            s = env.reset()
            for step in range(STEPS):
                a = np.argmax(model.predict(s.reshape(-1, STATE))) if np.random.random() > epsilon else np.random.randint(4)
                #a = np.random.randint(2)
                s1, r, d = env.step(a)
                q.put((s, a, r, s1, d))
                rewards += r
                if d:
                    break
                s = s1
            MONITOR_Q.put(rewards)
            if rewards > BEST:
                BEST = rewards
                print("new best: {} ({})".format(BEST, n))


def train(model, q, gamma=0.99):
    """
    train the global model
    """
    X = np.zeros((q.qsize(), STATE))
    y = np.zeros((q.qsize(), 4))
    i = 0
    while not q.empty():
        s, a, r, s1, d = q.get()
        reward = r if d else r + gamma * np.amax(model.predict(s1.reshape(-1, STATE)))
        target = model.predict(s.reshape(-1, STATE)).reshape(4)
        target[a] = reward
        X[i] = s
        y[i] = target
        i += 1
    model.fit(X, y, epochs=1, verbose=0)
    return model
"""
def train(model, target_model, q, gamma=0.99):
    X = np.zeros((64, 10, 10))
    y = np.zeros((64, 4))
    i = 0
    while not q.empty():
        s, a, r, s1, d = q.get()
        reward = r if d else r + gamma * np.amax(target_model.predict(s1.reshape(-1, 10, 10, 1)))
        target = model.predict(s.reshape(-1, 10, 10, 1)).reshape(4)
        target[a] = reward
        X[i] = s
        y[i] = target
    model.fit(X.reshape(-1, 10, 10, 1), y.reshape(-1, 4), epochs=1, verbose=0)
    return model
"""

def test_model(model):
    #env = gym.make(ENV)
    for _ in range(5):
        s = env.reset()
        for step in range(STEPS):
            a = np.argmax(model.predict(s.reshape(-1, STATE)))
            s1, _, d, _ = env.step(a)
            env.render()
            if d:
                break
            s = s1


if __name__ == '__main__':
    global_model = build_model()
    global_model.load_weights("applefixedxy.h5")
    local_models = [build_model() for _ in range(THREADS)]
    env = AppleGrid(render=False)
    s = env.reset()
    global_model.fit(s.reshape(-1, STATE), np.array([.25, .25, .25, .25]).reshape(-1, 4), epochs=1, verbose=0)
    graph = tf.get_default_graph()
    q = Queue()
    epsilon = 0.2

    for episode in range(1000):
        play_threads = [threading.Thread(target=play, args=(local_model, graph, q, epsilon, n))
                        for n, local_model in enumerate(local_models)]
        for thread in play_threads:
            thread.start()
        for thread in play_threads:
            thread.join()
        global_model = train(global_model, q)
        for lm in local_models:
            lm.set_weights(global_model.get_weights())
        epsilon = np.max([epsilon * .995, 0.05])

        if episode % 5 == 0:
            print("episode {}".format(episode))
            ep_results = []
            while not MONITOR_Q.empty():
                ep_results.append(MONITOR_Q.get())
            print("episode average: {} ({})".format(round(np.mean(ep_results), 3), BEST))
        if episode % 25 == 0:
            print("epsilon:", epsilon)
            global_model.save_weights("applefixedxy.h5")
            #test_model(target_model)