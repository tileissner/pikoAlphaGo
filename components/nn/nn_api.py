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

    #data = pd.read_json('replaybuffer.json', lines=True)

    #%%
    """
    reads json objects from json file and converts them into numpy array
    """
    states = []
    move = []
    win = []

    with open('replaybufferFlat.json') as json_file:
        data = json.load(json_file)
        # print(type(data[0]))
        # print(data[0])
        for line in range(len(data)):
            #print(data[line])
            states.append(data[line]['state'])
            move.append(data[line]['probabilities'])
            win.append(data[line]['winner'])
        # for state in data['state']:
        #     print(state)

    ALL_STATES = np.stack(states, axis=0)
    WINNER = np.stack(win, axis=0)
    MOVES = np.array(move)

    ALL_STATES = ALL_STATES.reshape(ALL_STATES.shape[0], 5, 5, 1)
    # WINNER = WINNER.reshape(WINNER.shape[0], 5, 5, 1)
    #MOVES = MOVES.reshape(MOVES.shape[0], 5, 5, 1)

#%%
    ALL_STATES = np.float64(ALL_STATES)
    print(ALL_STATES.dtype)

    input_shape = ALL_STATES.shape

    #print("states: {}, winner: {}, moves: {} ".format(ALL_STATES.shape, WINNER.shape, move.shape))

    return MOVES, WINNER, ALL_STATES, input_shape


MOVES, WINNER, ALL_STATES, input_shape = load_data()

def create_net():
    net = nn_model.NeuralNetwork()
    net.compile(optimizer='sgd', loss=['mse', 'categorical_crossentropy'])
    net.build(input_shape)
    #net.summary()
    return net


net = create_net()


def train_model(net, features, labels):
    EPOCHS = 50
    print(EPOCHS, "test")
    net.fit(features, labels, epochs=EPOCHS)
    #test_loss, test_acc = model.evaluate(test)"""
    net.save('models/model')


train_model(net, ALL_STATES, [WINNER, MOVES])


def load_model():
    pass
    #return load_model("path_to_saved_model")