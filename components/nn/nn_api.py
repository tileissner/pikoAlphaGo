import datetime
import json
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers

import components.nn.nn_model as nn_model
from utils import constants
from split_input import split_input

class NetworkAPI:
    ALL_STATES = None
    WINNER = None
    MOVES = None

    input_shape = None

    net = None
    pathToModel = ""

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
        #with open(os.path.join(dirname, '../../replaybuffer_perfect_game_last3turns.json')) as json_file:
        with open(os.path.join(dirname, '../../replaybuffer.json')) as json_file:
            data = json.load(json_file)
            for line in range(len(data)):
                states.append(data[line]['state'])
                move.append(data[line]['probabilities'])
                win.append(data[line]['winner'])

        ALL_STATES = np.stack(states, axis=0)
        WINNER = np.stack(win, axis=0)
        MOVES = np.array(move)

        # wieder auf 5 ändern an der letzten stelle wenn parents in der predcition enthalten
        ALL_STATES = ALL_STATES.reshape(ALL_STATES.shape[0], constants.board_size, constants.board_size,
                                        constants.input_stack_size)
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

    def create_net(self, input_shape_param = None):
        # lr = 0.001 zu groß für sgd -> loss für probs geht auf ~360
        # Adam vs SGD -> erst mal mit SGD für "einfaches" debugging und für abschließende optimierung dann mit Adam
        # learning rate für sgd liegt zwischen 0.01 und 0.1 bei perfektem game buffer
        # bei random buffer kleiner ~0.0001
        self.optimizer = optimizers.SGD(learning_rate=0.01)
        self.net = nn_model.NeuralNetwork()
        self.net.compile(optimizer=self.optimizer,
                         loss=['mse', 'categorical_crossentropy'])  # l2 regularization evtl hinzufügen
        if input_shape_param is not None:
            self.input_shape = input_shape_param

        self.net.build(self.input_shape)


    def train_model(self, features, labels):
        EPOCHS = constants.epochs

        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        checkpoint_path = "training_1/{epoch:04d}-cp.ckpt"
        # checkpoint_dir = os.path.dirname(checkpoint_path)

        # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
        hist_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=self.log_weights(EPOCHS), log_dir=log_dir)
        # neue funktion hier -> checkpoints erstellen
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True,
                                                         save_freq="epoch", period=1, verbose=1)
        self.net.fit(features, labels, epochs=EPOCHS, batch_size=constants.custom_batch_size, callbacks=[hist_callback, cp_callback])

    def save_model(self, filename=None):
        ############hier decouplen damit wir auch ohne training ein netz speichern können direkt nach initalisierung
        dirname = os.path.dirname(__file__)
        if filename is None:
            self.pathToModel = 'models/model' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        else:
            self.pathToModel = 'models/model' + filename
        constants.challengerNetFileName = self.pathToModel
        self.net.save(os.path.join(dirname, self.pathToModel))
        return self.pathToModel

    # %%
    def log_weights(self, epochs):
        writer = tf.summary.create_file_writer("/tmp/mylogs/eager")

        with writer.as_default():
            for tf_var in self.net.trainable_weights:
                tf.summary.histogram(tf_var.name, tf_var.numpy(), step=epochs)

    def model_load(self, pathToFile=None):
        if pathToFile is None:
            dirname = os.path.dirname(__file__)
            filename = os.path.join(dirname, 'models/model/saved_model.pb')
        else:
            dirname = os.path.dirname(__file__)
            # filename = os.path.join(dirname, pathToFile + 'saved_model.pb')
            self.net = tf.keras.models.load_model(os.path.join(dirname, pathToFile))
            self.pathToModel = pathToFile

    def getPredictionFromNN(self, state, parentStates, color):

        # TODO ergibt keinen sinn, fix für "startspieler", wenn color = None ist
        if color is None:
            WHITE, NONE, BLACK = range(-1, 2)
            color = BLACK

        colorArray = None
        if color == -1:
            colorArray = np.ones((constants.board_size, constants.board_size), dtype=int)
        elif color == 1:
            colorArray = np.zeros((constants.board_size, constants.board_size), dtype=int)

        missingStates = constants.state_history_length - len(parentStates)

        emptyState = np.zeros((constants.board_size, constants.board_size), dtype=int)
        for i in range(0, missingStates):
            parentStates.append(emptyState)

        inputList = []
        inputList.append(state)
        for elem in parentStates:
            inputList.append(elem)
        inputList.append(colorArray)

        # state = np.array(state)
        # state = np.float64(state)
        # state = state.reshape(1, 5, 5, 1)

        inputList = np.array(inputList)
        inputList = np.float64(inputList) #ein input stack (als array)
        inputList = split_input(inputList)
        inputList = inputList.reshape(1, constants.board_size, constants.board_size, constants.input_stack_size)

        winner, probs = self.net.predict(inputList)
        winner = winner.item(0)
        probs = probs.flatten().tolist()
        # print("winner: {} , probs: {}".format(winner, probs))
        return winner, probs
