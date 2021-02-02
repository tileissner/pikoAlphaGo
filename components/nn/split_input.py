import numpy as np
from utils import constants


def split_input(state_history):
    full_input = np.zeros(shape=(constants.input_stack_size, constants.board_size, constants.board_size), dtype=np.float64)

    for i in range(constants.input_states):
        whitemask = np.zeros(shape=(constants.board_size, constants.board_size), dtype=int)
        blackmask = np.zeros(shape=(constants.board_size, constants.board_size), dtype=int)
        for j in range(constants.board_size):
            for h in range(constants.board_size):
                if state_history[i, j, h] == 1:
                    blackmask[j, h] = 1
                elif state_history[i, j, h] == -1:
                    whitemask[j, h] = 1
                #print("{} zu {} in schwarz und zu {} in weiß gemacht".format(nplist[i,j,h],blackmask[j,h],whitemask[j,h]))
                full_input[i] = blackmask
                full_input[i+constants.state_history_length+1] = whitemask

    full_input[constants.input_states*2] = state_history[constants.state_history_length+1]
    return full_input