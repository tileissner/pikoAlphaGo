import json
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers

import components.nn.nn_model as nn_model
import datetime


class NetworkAPI:
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
        """

        states = []
        move = []
        win = []

        dirname = os.path.dirname(__file__)
        with open(os.path.join(dirname, '../../replaybuffer.json')) as json_file:
            data = json.load(json_file)
            for line in range(len(data)):
                states.append(data[line]['state'])
                move.append(data[line]['probabilities'])
                win.append(data[line]['winner'])

        ALL_STATES = np.stack(states, axis=0)
        WINNER = np.stack(win, axis=0)
        MOVES = np.array(move)

        ALL_STATES = ALL_STATES.reshape(ALL_STATES.shape[0], 5, 5, 1)
        # WINNER = WINNER.reshape(WINNER.shape[0], 5, 5, 1)
        # MOVES = MOVES.reshape(MOVES.shape[0], 5, 5, 1)

        # %%
        ALL_STATES = np.float64(ALL_STATES)
        print(ALL_STATES.dtype)

        input_shape = ALL_STATES.shape

        self.MOVES = MOVES
        self.WINNER = WINNER
        self.ALL_STATES = ALL_STATES
        self.input_shape = input_shape

    def create_net(self):
        #lr = 0.001 zu groß für sgd -> loss für probs geht auf ~360
        #Adam vs SGD -> erst mal mit SGD für "einfaches" debugging und für abschließende optimierung dann mit Adam
        #learning rate für sgd liegt zwischen 0.01 und 0.1 bei perfektem game buffer
        #bei random buffer kleiner ~0.0001
        self.optimizer = optimizers.SGD(learning_rate=0.0001)
        self.net = nn_model.NeuralNetwork()
        self.net.compile(optimizer=self.optimizer, loss=['mse', 'categorical_crossentropy'])
        self.net.build(self.input_shape)

    def train_model(self, features, labels):
        EPOCHS = 300

        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
        hist_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=self.log_weights(EPOCHS), log_dir=log_dir)
        self.net.fit(features, labels, epochs=EPOCHS, callbacks=[hist_callback])

        # test_loss, test_acc = model.evaluate(test)"""
        dirname = os.path.dirname(__file__)
        self.net.save(os.path.join(dirname, 'models/model/'))


    # %%
    def log_weights(self, epochs):
            writer = tf.summary.create_file_writer("/tmp/mylogs/eager")

            with writer.as_default():
                for tf_var in self.net.trainable_weights:
                    tf.summary.histogram(tf_var.name, tf_var.numpy(), step=epochs)


    def model_load(self, fileName = None):
        if fileName is None:
            dirname = os.path.dirname(__file__)
            filename = os.path.join(dirname, 'models/model/saved_model.pb')
        else:
            dirname = os.path.dirname(__file__)
            filename = os.path.join(dirname, 'models/model/' + fileName + '.pb')

        self.net = tf.keras.models.load_model(os.path.join(dirname, 'models/model/'))

    def getPredictionFromNN(self, state):
        state = np.array(state)
        state = np.float64(state)
        state = state.reshape(1, 5, 5, 1)

        winner, probs = self.net.predict(state)
        winner = winner.item(0)
        probs = probs.flatten().tolist()
        #print("winner: {} , probs: {}".format(winner, probs))
        return winner, probs

