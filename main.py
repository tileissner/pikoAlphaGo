import os
import sys
import threading
import time

import components.go.goEngineApi as goApi
import components.nn.nn_api as nn_api
import utils.constants as constants
import utils.readConfigFile as configFile
import numpy as np


BLACK, NONE, WHITE = range(-1, 2)

class selfplay:
    trainingSetList = []

    winsblack = 0
    winswhite = 0

    def __init__(self):
        print("in_init")
        self.winsblack = 0
        self.second = 0

    def startSelfPlayGame(self, file_name, board_size, color):
        # selfplay test with hardcoded beginner (black)
        trainingSet = goApi.selfPlay(board_size, color)
        self.trainingSetList.append(trainingSet)

        # if trainingSet[-1].winner == -1:
        #     self.winsblack += 1
        # else:
        #     self.winswhite += 1
        #
        # with open(file_name, 'a', 1) as f:
        #     index = 0
        #     for t in trainingSet:
        #         # f.write(np.array2string(t.state) + t.probabilities + t.winner + t.color + os.linesep)
        #         # t.getAsJSON()
        #         # f.write(np.array2string(t.state) + str(t.winner) + str(t.color) + os.linesep)
        #         if index == (len(trainingSet) - 1):
        #             f.write(t.getAsJSON(True) + "\n")
        #         else:
        #             f.write(t.getAsJSON(False) + "\n")
        #         index += 1

    def startSelfPlay(self, thread_count, board_size, color):
        file_name = "replaybuffer.json"
        threads = []
        for i in range(0, thread_count):
            # target = name for method that must be executed in thread
            threads.append(
                threading.Thread(target=self.startSelfPlayGame, args=(file_name, board_size, color)))
            threads[-1].start()
        for t in threads:
            t.join()
        #hier wartet er auf alle threads bis sie feritg sind

        # # check what the heck the file had
        # uniq_lines = set()
        # with open('replaybuffer.json', 'r') as f:
        #     for l in f:
        #         uniq_lines.add(l)
        # # for u in uniq_lines:
        # # sys.stdout.write(u)

    def writeTrainingSetsToJsonBuffer(self):
        listIndex = 0
        with open('replaybuffer.json', 'a', 1) as f:
            f.write("[")
            #1 traininget = 1 entire game
            for trainingSet in self.trainingSetList:
                rowIndex = 0
                #1 t = eine zeile (=1 Zustand) des Spiels
                for t in trainingSet:
                    if listIndex == (len(self.trainingSetList) - 1) and rowIndex == (len(trainingSet) -1):
                        f.write(writeHistoryStates(trainingSet, rowIndex, True))
                    else:
                        f.write(writeHistoryStates(trainingSet, rowIndex, False))
                        # if rowIndex == 0:
                        #     f.write(writeHistoryStates(trainingSet, 0, False))
                        # if rowIndex == 1:
                        #     f.write(writeHistoryStates(trainingSet, 1, False))
                        # if rowIndex == 2:
                        #     f.write(writeHistoryStates(trainingSet, 2, False))
                        # if rowIndex == 3:
                        #     f.write(writeHistoryStates(trainingSet, 3, False))
                    rowIndex += 1
                listIndex += 1
            f.write("]")

# main function of all sub-programs
def main(args):
    print(args)
    start = time.time()
    if (len(sys.argv)) > 1:
        constants.configFileLocation = args[1]
    else:
        constants.configFileLocation = "config.yaml"
    # read config file and store it in constants.py
    configFile.readConfigFile(constants.configFileLocation)
    if os.path.exists("replaybuffer.json"):
        os.remove("replaybuffer.json")

    # with open("replaybuffer.json", 'a') as f:
    #     f.write("[")

    BLACK, NONE, WHITE = range(-1, 2)
    sp = selfplay()
    sp.startSelfPlay(constants.thread_count, constants.board_size, BLACK)
    #wait for finish of threads
    sp.writeTrainingSetsToJsonBuffer()

    # warte auf threads
    # sp.startSelfPlay(constants.thread_count, constants.board_size, BLACK)

    # append eckige klammern, damit gültiges json
    # with open("replaybuffer.json", 'a') as f:
    #     f.write("]")

    print("White: ", sp.winswhite)
    print("Black: ", sp.winsblack)

    net_api = nn_api.NetworkAPI()
    net_api.load_data()  # werte initialisiert
    net_api.create_net()
    net_api.train_model(net_api.ALL_STATES, [net_api.WINNER, net_api.MOVES])

    end = time.time()
    print("Time elapsed: ", end - start)


# startSelfPlay(constants.thread_count, constants.board_size, BLACK)


def writeHistoryStates(trainingSet, index, lastElement):
    previousStates = []
    # dient auch als leeres board wenn wir auffüllen müssen
    # falls nciht genug vorherige zustände vorhanden sind
    color_w = np.zeros((5,5), dtype=int)
    color_b = np.ones((5,5), dtype=int)
    counter = 0

    startValue = index - constants.state_history_length
    if startValue < 0:
        startValue = 0

    #fall 0: geht er dann überhaupt rein?
    #for i in range(startValue, index):
    for i in range(startValue, index):
        previousStates.append(trainingSet[i].state)
        counter = counter + 1

    previousStates = previousStates[::-1]

    toBeFilled = constants.state_history_length - counter
    previousStatesFilled = []
    for i in range(0, toBeFilled):
        previousStatesFilled.append(np.zeros((5,5), dtype=int))

    if trainingSet[index].color == BLACK:
        previousStatesFilled.append(color_b)
    else:
        previousStatesFilled.append(color_w)



    # die fehlenden 0er listen als history auffuellen -> gleiche anzahl an history in jedem state (auch zb  im ersten state)
    for elem in previousStatesFilled:
        previousStates.append(elem)

    return trainingSet[index].getAsJSONWithPreviousStates(lastElement, previousStates) + "\n"




if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv)



