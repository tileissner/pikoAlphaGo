#%%
import numpy as np

import utils.readConfigFile as configFile
from components.nn.nn_api import NetworkAPI
from utils import constants

constants.configFileLocation = "/home/tim/Documents/uni/WS20/alphago/pikoAlphaGo/config.yaml"
# read config file and store it in constants.py
configFile.readConfigFile(constants.configFileLocation)

all_states = []
empty_board = [[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]]
zug_acht = [[[0, 0, 0, 0, 0], [0, -1, -1, 1, 0], [0, 1, 1, 0, 0], [0, -1, 1, 0, 0], [0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0], [0, -1, -1, 0, 0], [0, 1, 1, 0, 0], [0, -1, 1, 0, 0], [0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0], [0, -1, 0, 0, 0], [0, 1, 1, 0, 0], [0, -1, 1, 0, 0], [0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0], [0, -1, 0, 0, 0], [0, 0, 1, 0, 0], [0, -1, 1, 0, 0], [0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]]
zug_sechzehn = [[[0, 0, 0, -1, 0], [0, -1, -1, 1, 0], [-1, 1, 1, 1, 0], [0, -1, 1, 0, -1], [-1, 1, 1, 0, 0]], [[0, 0, 0, -1, 0], [0, -1, -1, 1, 0], [-1, 1, 1, 1, 0], [0, -1, 1, 0, -1], [-1, 0, 1, 0, 0]], [[0, 0, 0, -1, 0], [0, -1, -1, 1, 0], [-1, 1, 1, 1, 0], [1, -1, 1, 0, -1], [0, 0, 1, 0, 0]], [[0, 0, 0, -1, 0], [0, -1, -1, 1, 0], [-1, 1, 1, 1, 0], [0, -1, 1, 0, -1], [0, 0, 1, 0, 0]], [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]]
zug_22 = [[[0, 1, 1, -1, 0], [1, 0, 0, 1, 0], [0, 1, 1, 1, 0], [-1, 0, 1, -1, -1], [0, 1, 1, 0, 0]], [[0, 1, 0, -1, 0], [1, -1, -1, 1, 0], [0, 1, 1, 1, 0], [-1, 0, 1, -1, -1], [0, 1, 1, 0, 0]], [[0, 1, 0, -1, 0], [1, -1, -1, 1, 0], [0, 1, 1, 1, 0], [-1, 0, 1, 0, -1], [0, 1, 1, 0, 0]], [[0, 0, 0, -1, 0], [1, -1, -1, 1, 0], [0, 1, 1, 1, 0], [-1, 0, 1, 0, -1], [0, 1, 1, 0, 0]], [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]]

all_states.append(empty_board)
all_states.append(zug_acht)
all_states.append(zug_sechzehn)
all_states.append(zug_22)

def getPredictionFromNN(net, state):
    state = np.array(state)
    state = np.float64(state)
    state = state.reshape(1, constants.board_size, constants.board_size, constants.input_stack_size)

    winner, probs = net.predict(state)
    winner = winner.item(0)
    probs = probs.flatten().tolist()
    # print("winner: {} , probs: {}".format(winner, probs))
    return winner, probs

net_api = NetworkAPI()
net_api.load_data()
net_api.create_net()

net_api.train_model(net_api.ALL_STATES, [net_api.WINNER, net_api.MOVES])
fitted = net_api.save_model()
print(fitted)

net_api.create_net()

#prediction mit frisch initialisiertem Netz, aka ohne Training
for state in all_states:
    winner_nt, probs_nt = getPredictionFromNN(net_api.net, state)
    state_reshape = np.reshape(state, (5,5,5))
    probs_reshape = probs_nt[:-1]
    probs_reshape = np.reshape(probs_reshape, (5,5))
    print('state + history + color matrix, in order from t to t-3:\n{}'.format(state_reshape))
    print('probs from untrained net:\n{}, pass prob {}, winner: {}'.format(probs_reshape, probs_nt[-1:], winner_nt))

#Gewichte nach einer Epoche laden
net_api.net.load_weights("/home/tim/Documents/uni/WS20/alphago/pikoAlphaGo/components/nn/training_1/0001-cp.ckpt")
for state in all_states:
    winner_nt, probs_nt = getPredictionFromNN(net_api.net, state)
    state_reshape = np.reshape(state, (5,5,5))
    probs_reshape = probs_nt[:-1]
    probs_reshape = np.reshape(probs_reshape, (5,5))
    print('state + history + color matrix, in order from t to t-3:\n{}'.format(state_reshape))
    print('probs after training for 1 epoch:\n{}, pass prob {}, winner: {}'.format(probs_reshape, probs_nt[-1:], winner_nt))

#nach 100 epochen
net_api.net.load_weights("/home/tim/Documents/uni/WS20/alphago/pikoAlphaGo/components/nn/training_1/0100-cp.ckpt")
for state in all_states:
    winner_nt, probs_nt = getPredictionFromNN(net_api.net, state)
    state_reshape = np.reshape(state, (5,5,5))
    probs_reshape = probs_nt[:-1]
    probs_reshape = np.reshape(probs_reshape, (5,5))
    print('state + history + color matrix, in order from t to t-3:\n{}'.format(state_reshape))
    print('probs after training for 100 epochs:\n{}, pass prob {}, winner: {}'.format(probs_reshape, probs_nt[-1:], winner_nt))

#Gewichte nach 300 Epochen laden
net_api.net.load_weights("/home/tim/Documents/uni/WS20/alphago/pikoAlphaGo/components/nn/training_1/0300-cp.ckpt")
for state in all_states:
    winner_nt, probs_nt = getPredictionFromNN(net_api.net, state)
    state_reshape = np.reshape(state, (5,5,5))
    probs_reshape = probs_nt[:-1]
    probs_reshape = np.reshape(probs_reshape, (5,5))
    print('state + history + color matrix, in order from t to t-3:\n{}'.format(state_reshape))
    print('probs after training for 300 epochs:\n{}, pass prob: {}, winner: {}'.format(probs_reshape, probs_nt[-1:], winner_nt))

"""
net_api.net.load_weights("/home/tim/Documents/uni/WS20/alphago/pikoAlphaGo/components/nn/training_1/0700-cp.ckpt")
for state in all_states:
    winner_nt, probs_nt = getPredictionFromNN(net_api.net, state)
    state_reshape = np.reshape(state, (5,5,5))
    probs_reshape = probs_nt[:-1]
    probs_reshape = np.reshape(probs_reshape, (5,5))
    print('state + history + color matrix, in order from t to t-3:\n{}'.format(state_reshape))
    print('probs after training for 700 epochs:\n{}, pass prob: {}, winner: {}'.format(probs_reshape, probs_nt[-1:], winner_nt))
"""
#%%
"""
net_api.model_load("models/model20210118-155808")

for state in all_states:
    winner, probs = getPredictionFromNN(net_api.net, state)
    print('probs from net: {}, winner: {}'.format(probs, winner))
    print('sum of probs: {}'.format(sum(probs)))
    state_reshape = np.reshape(state, (5,5,5))
    probs_reshape = probs[:-1]
    probs_reshape = np.reshape(probs_reshape, (5,5))
    print(probs_reshape)


zugnull=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
zugacht=[0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
zug16=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
zug22=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]

zugnull = zugnull[:-1]
zugacht = zugacht[:-1]
zug16 = zug16[:-1]
zug22 = zug22[:-1]

zugnull= np.reshape(zugnull, (5,5))
zugacht= np.reshape(zugacht, (5,5))
zug16= np.reshape(zug16,(5,5))
zug22= np.reshape(zug22,(5,5))

print(zugnull, zugacht, zug16, zug22)
"""