from components.go import coords
from components.go.goEngineApi import choseActionAccordingToMCTS, createGame
from components.nn.nn_api import NetworkAPI
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers


def getPredictionFromNN(self, state):
    state = np.array(state)
    state = np.float64(state)
    state = state.reshape(1, 5, 5, 5)

    winner, probs = self.net.predict(state)
    winner = winner.item(0)
    probs = probs.flatten().tolist()
    # print("winner: {} , probs: {}".format(winner, probs))
    return winner, probs


net_api = NetworkAPI()

net_api.load_data()
net_api.create_net()
net_api.train_model(net_api.ALL_STATES, [net_api.WINNER, net_api.MOVES])

net_api.model_load()

pos = createGame(5, 1)
pos = pos.play_move(coords.from_flat(5), 1, False)
pos = pos.play_move(coords.from_flat(7), -1, False)
print(pos.board)
winner, probs = net_api.getPredictionFromNN(pos.board)

print(winner)
print(probs)




# tf.random_normal_initializer(
#     mean=0.0, stddev=0.05, seed=None
# )

#net_api.model_load()

