import sys
import threading
import time

import components.go.goEngineApi as goApi
import components.nn.nn_api as nn_api
import utils.constants as constants
import utils.readConfigFile as configFile


class selfplay:
    winsblack = 0
    winswhite = 0

    def __init__(self):
        print("in_init")
        self.winsblack = 0
        self.second = 0

    def startSelfPlayGame(self, file_name, board_size, color):
        # selfplay test with hardcoded beginner (black)
        trainingSet = goApi.selfPlay(board_size, color)

        if trainingSet[-1].winner == -1:
            self.winsblack += 1
        else:
            self.winswhite += 1

        with open(file_name, 'a', 1) as f:
            index = 0
            for t in trainingSet:
                # f.write(np.array2string(t.state) + t.probabilities + t.winner + t.color + os.linesep)
                # t.getAsJSON()
                # f.write(np.array2string(t.state) + str(t.winner) + str(t.color) + os.linesep)
                if index == (len(trainingSet) - 1):
                    f.write(t.getAsJSON(True) + "\n")
                else:
                    f.write(t.getAsJSON(False) + "\n")
                index += 1

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
        # check what the heck the file had
        uniq_lines = set()
        with open('replaybuffer.json', 'r') as f:
            for l in f:
                uniq_lines.add(l)
        # for u in uniq_lines:
        # sys.stdout.write(u)


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

    with open("replaybuffer.json", 'a') as f:
        f.write("[")

    BLACK, NONE, WHITE = range(-1, 2)
    sp = selfplay()
    sp.startSelfPlay(constants.thread_count, constants.board_size, BLACK)
    # sp.startSelfPlay(constants.thread_count, constants.board_size, BLACK)

    # append eckige klammern, damit gültiges json
    with open("replaybuffer.json", 'a') as f:
        f.write("]")

    print("White: ", sp.winswhite)
    print("Black: ", sp.winsblack)

    net_api = nn_api.NetworkAPI()
    net_api.load_data()  # werte initialisiert
    net_api.create_net()
    net_api.train_model(net_api.ALL_STATES, [net_api.WINNER, net_api.MOVES])

    end = time.time()
    print("Time elapsed: ", end - start)


# startSelfPlay(constants.thread_count, constants.board_size, BLACK)


if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv)
