import os
import sys
import threading
import time

import components.go.goEngineApi as goApi
import components.nn.nn_api as nn_api
import utils.constants as constants
import utils.readConfigFile as configFile
import numpy as np


WHITE, NONE, BLACK = range(-1, 2)

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

        if trainingSet[-1].winner == -1:
            self.winsblack += 1
        else:
            self.winswhite += 1
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

    def writeTrainingSetsToJsonBuffer(self, iteration):
        listIndex = 0
        with open('replaybuffer.json', 'a', 1) as f:
            if iteration == 0:
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

    sys.setrecursionlimit(100000)
    print("recursion limit: ", sys.getrecursionlimit())

    # with open("replaybuffer.json", 'a') as f:
    #     f.write("[")

    WHITE, NONE, BLACK = range(-1, 2)

    #untrainiertes netz laden
    untrained_net = nn_api.NetworkAPI()
    #hack, damit wir subclassed model speichern können
    #man muss subclassed model nämlich erst auf irgendeine art & weise verwenden
    #die input_shape muss hier auch manuell gesetzt werden, da wir die daten noch nicht laden, da noch kein replaybuffer exisitiert
    #diese input_shape wird allerdings erst übernommen wenn das model genutzt wurde
    #wir lassen also predicten mit dummy variablen damit die input_shape ans netz übergeben wird
    #sobald diese shape übergeben wurde kann das untrainierte netz gespeichert werden
    initial_input_shape = (1, constants.board_size, constants.board_size, constants.input_stack_size)
    untrained_net.create_net(initial_input_shape)
    dummy_state = [[0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    winner_test, probs_test = untrained_net.getPredictionFromNN(dummy_state, [], BLACK)
    constants.currentBestNetFileName = untrained_net.save_model("untrained")

    for i in range(0, constants.pipeline_runs):

        # load old (=current best network)
        print("Lade bisheriges Netzwerk")
        currentBestNetApi = nn_api.NetworkAPI()
        currentBestNetApi.model_load(constants.currentBestNetFileName)

        print("start self-play with old current best network {}".format(constants.currentBestNetFileName))
        sp = selfplay()
        # starts the self play games with the current best network
        sp.startSelfPlay(constants.thread_count, constants.board_size, BLACK)
        #wait for finish of threads
        sp.writeTrainingSetsToJsonBuffer(i)

        # now evaluation comes into play -> play as good as possible
        constants.competitive = 1

        # warte auf threads
        # sp.startSelfPlay(constants.thread_count, constants.board_size, BLACK)

        # append eckige klammern, damit gültiges json
        # with open("replaybuffer.json", 'a') as f:
        #     f.write("]")

        print("White: ", sp.winswhite)
        print("Black: ", sp.winsblack)

        challengerNetApi = nn_api.NetworkAPI()
        challengerNetApi.load_data()  # werte initialisiert
        challengerNetApi.net = currentBestNetApi.net
        winner_loaded_test, probs_loaded_test = challengerNetApi.getPredictionFromNN(dummy_state, [], BLACK)
        #challengerNetApi.create_net()
        print("Trainiere neues Netzwerk")
        challengerNetApi.train_model(challengerNetApi.ALL_STATES, [challengerNetApi.WINNER, challengerNetApi.MOVES])
        winner_loaded_trained, probs_loaded_trained = challengerNetApi.getPredictionFromNN(dummy_state, [], BLACK)
        challengerNetApi.save_model()
        # neues netz wurde erstellt -> wird zum challenger netz

        print("Evaluiere neues Netzwerk")
        print(constants.currentBestNetFileName)
        print(constants.challengerNetFileName)

        # reset competitive for self play exploration
        constants.competitive = 0


        goApi.evaluateNet(constants.board_size, -1, constants.currentBestNetFileName, constants.challengerNetFileName)

        print("Neues netz: " + constants.currentBestNetFileName)
        removeLastCharacter('replaybuffer.json')

        # wenn neues netz besser -> kopiere weights des neuen netzes

    print("Bestes Netz am Ende aller Pipeline Runs: {}".format(constants.currentBestNetFileName))
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


def removeLastCharacter(filename):
    f = open(filename, "r")
    content = f.read()
    content = content[:-1]
    content = content + ","



    f = open(filename, "w")
    f.write(content)
    f.close()



if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv)


