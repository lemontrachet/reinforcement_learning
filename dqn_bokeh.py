import numpy as np
from bokeh.io import curdoc
from bokeh.models import ColumnDataSource
from bokeh.plotting import Figure
from collections import deque
import gym
import time
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.optimizers import RMSprop
from sklearn.preprocessing import StandardScaler


def reward(state, diff):
    l1, pos, _ = state
    c = l1 + 0.5
    if abs(pos - c) < 0.25:
        return 2
    if abs(pos - c) < 0.5:
        return 1
    if diff < 0:
        return 0.2
    else:
        return diff * -0.1


def build_model():
    model = Sequential([
        Dense(128, input_shape=(3,), activation="elu"),
        Dense(128, activation="elu"),
        Dense(3, activation="linear")
    ])
    model.compile(optimizer=RMSprop(lr=0.001, clipvalue=1.0),
                  loss="mse")
    return model


def train_model(model, target_model, replay_buffer):
    gamma = 0.9 # value of future rewards
    batch_size = 32
    if len(replay_buffer) < batch_size:
        return model

    # take a random sample from replay buffer for training
    memory = np.array(replay_buffer)[np.random.choice(len(replay_buffer), batch_size, replace=False), :]

    X, y = np.zeros((batch_size, 3)), np.zeros((batch_size, 3))
    for i in range(len(memory)):
        s = memory[i, 0]
        a = memory[i, 1]
        r = memory[i, 2]
        s1 = memory[i, 3]
        total_r = r + gamma * np.amax(target_model.predict(np.array(s1).reshape(-1, 3)))
        X[i] = s
        y[i, int(a)] = total_r

    model.fit(X, y, epochs=1, verbose=0)
    return model


def update():
    global n, av, model, target_model, epsilon, replay_buffer, state, turn, rewards
    actions = {0: -0.5, 1: 0., 2: 0.5}

    # save the current position
    av.append(state[1])

    # get action
    a = (np.argmax(target_model.predict(np.array(state).reshape(-1, 3))) if np.random.random() > epsilon
            else np.random.randint(3))
    action_value = actions[a]

    # apply action and get new state
    state1 = state.copy()
    state1[1] = np.max([-10, np.min([10, state1[1] + action_value])])

    # get reward
    diff = abs(state1[1] - (state1[0] + state[2]) / 2) - abs(state[1] - (state[0] + state[2]) / 2)
    r = reward(state1, diff)
    rewards.append(r)
    if n % 1000 == 0:
        print("av. rewards: {} - epsilon: {}".format(np.mean(rewards), epsilon))

    # add memory to the buffer and train
    replay_buffer.append((state, a, r, state1))
    model = train_model(model, target_model, replay_buffer)

    # transfer weights to target model
    if n % 10 == 0:
        target_model.set_weights(model.get_weights())

    # update the plot
    new_data = dict(x=[n], x_lag=[n - 50], l1=[state[0]], pos=[state[1]], l2=[state[2]], av0=[np.mean(av)])

    n += 1

    # reduce the exploration rate
    epsilon = np.max([0.05, epsilon - 0.00005])

    # turn the lanes
    if state[0] > 5:
        turn = -1
    elif state[2] < 1:
        turn = 1
    elif np.random.random() > 0.98:
        turn = np.random.choice([-1, 0, 1])
    state1[0] += turn / 100
    state1[2] += turn / 100

    state = state1
    source.stream(new_data, 200)


#if __name__ == '__main__':

# globals - for bokeh plotting
n = 50
model, target_model = build_model(), build_model()
epsilon = 1 # initial exploration rate

av, rewards = deque(maxlen=100), deque(maxlen=100)
replay_buffer = deque(maxlen=1000000)
state = [4, 0, 5] # initial state: left lane, position, right lane
turn = 0

source = ColumnDataSource(dict(x=[], x_lag=[], l1=[], pos=[], l2=[], av0=[]))

fig = Figure(plot_width=1200, plot_height=400)
fig.line(source=source, x="x", y="l1", line_width=2, alpha=.85, color="red")
fig.line(source=source, x="x", y="l2", line_width=2, alpha=.85, color="red")
fig.line(source=source, x="x_lag", y="av0", line_width=2, alpha=.85, color="green")

curdoc().add_root(fig)
curdoc().add_periodic_callback(update, 50)