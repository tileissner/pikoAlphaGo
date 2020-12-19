import os

import pandas as pd
import numpy as np
import json
import tensorflow as tf


import components.nn.nn_model as nn_model
#import nn_model

class NetworkAPI():

    ALL_STATES = None
    WINNER = None
    MOVES = None

    input_shape = None

    net = None

    def load_data(self):
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

        dirname = os.path.dirname(__file__)
        with open(os.path.join(dirname, '../../replaybuffer.json')) as json_file:
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
        self.MOVES = MOVES
        self.WINNER = WINNER
        self.ALL_STATES = ALL_STATES
        self.input_shape = input_shape


    def create_net(self):
        self.net = nn_model.NeuralNetwork()
        self.net.compile(optimizer='sgd', loss=['mse', 'categorical_crossentropy'])
        self.net.build(self.input_shape)
        #net.summary()

    def train_model(self, features, labels):
        EPOCHS = 50
        print(EPOCHS, "test")
        self.net.fit(features, labels, epochs=EPOCHS)
        #test_loss, test_acc = model.evaluate(test)"""
        #self.net.save('models/model')
        dirname = os.path.dirname(__file__)
        self.net.save(os.path.join(dirname, 'models/model/'))

    #%%
    # def model_load(self, path):
    #     self.net = tf.keras.models.load_model(path)
    #     #self.net = tf.keras.models.load_model('models/model')
    def model_load(self):
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, 'models/model/saved_model.pb')

        self.net = tf.keras.models.load_model(os.path.join(dirname, 'models/model/'))

    def getPredictionFromNN(self, state):
        #print(type(self.net))
        #self.net.summary()
        #TODO richtige werte muessen hier rein
        state = np.array(state)
        state = np.float64(state)
        state = state.reshape(1,5,5,1)

        winner, probs = self.net.predict(state)
        #print("winner: {} , probs: {}".format(winner, probs))
        return winner, probs

    #test_state = [[-1, -1, -1, -1, -1], [-1, -1, -1, 0, 0], [-1, -1, -1, -1, 1], [-1, 1, 0, -1, 0], [-1, -1, -1, -1, 0]]
    #test_state = [[-1, -1, -1, -1, -1], [1, -1, -1, -1, 0], [1, 1, -1, -1, -1], [1, 1, 1, -1, 1], [0, 1, 1, -1, 0]]
    # test_state = [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 0]]
    # test_state = np.array(test_state)
    # print(test_state.shape)
    # test_state = test_state.reshape(1, 5, 5, 1)
    # test_state = np.float64(test_state)
    # print(test_state.shape)
    # print(test_state)

