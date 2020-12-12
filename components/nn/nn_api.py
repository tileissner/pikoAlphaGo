import pandas as pd
import numpy as np
import json

import nn_model

def load_data():
    """
    features in unserem fall: board state
    targets: move probs; value (für winner) also -1, 1

    feature_names = column_names[:-1]
    label_name = column_names[-1]


    #evtl format ändern für move, je nachdem ob cross-entropy mit jetzigem format klar kommt
    #my guess: prolly not, sollte wsl einfach flat array sein, sodass index der move zahl entspricht
    """

    data = pd.read_json('replaybuffer.json', lines=True)

    #%%
    """
    reads json objects from json file and converts them into numpy array
    """
    states = []
    move = []
    win = []
    for line in open("replaybuffer.json", "r"):
        line_ = json.loads(line)

        states_extract = np.asarray(line_["state"])
        states.append(states_extract)

        move_extract = np.asarray(line_["probabilities"])
        move.append(move_extract)

        win_extract = np.asarray(line_["winner"])
        win.append(win_extract)

    ALL_STATES = np.stack(states, axis=0)
    WINNER = np.stack(win, axis=0)
    MOVES = np.stack(move, axis=0)

    #print("states: {}, winner: {}, moves: {} ".format(ALL_STATES.shape, WINNER.shape, MOVES.shape))

    """reshape to fit tf input requirements"""
    ALL_STATES = ALL_STATES.reshape(ALL_STATES.shape[0], 5, 5, 1)
    #WINNER = WINNER.reshape(WINNER.shape[0], 5, 5, 1)
    #MOVES = MOVES.reshape(MOVES.shape[0], 5, 5, 1)

    input_shape = ALL_STATES.shape
    """
    evtl aufpassen wegen zuordnung, index der features und targets
    sollte aber theoretisch klappen
    """
#%%
    return MOVES, WINNER, ALL_STATES, input_shape

#MOVE, WINNER, FEATURES = load_data()

_, _, _, input_shape = load_data()
print(input_shape)

def create_net():
    net = nn_model.NeuralNetwork()
    net.compile(optimizer='sgd', loss=['mse', 'categorical_cross_entropy'])
    net.build(input_shape)
    net.summary()
    return net

net = create_net()

def train_model(net, features, labels):
    EPOCHS = 10
    # net.compile(optimizer='sgd', loss=['mse', 'categorical_cross_entropy'])
    print(EPOCHS, "test")
    net.fit(features, labels, epochs=EPOCHS)
    #test_loss, test_acc = model.evaluate(test)"""


def save_model(self):
    pass
#net.train_model(net, FEATURES, WINNER)


#model = nn_model.create_model()
#nn_model.train_model(model)