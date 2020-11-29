import os
import threading

import utils.readConfigFile as configFile
import components.go.goEngineApi as goApi
import utils.constants as constants
import sys
import numpy as np


BLACK, NONE, WHITE = range(-1, 2)


def startSelfPlayGame(file_name, board_size, color):
    # selfplay test with hardcoded beginner (black)
    trainingSet = goApi.selfPlay(board_size, color)
    with open(file_name, 'a', 1) as f:
        for t in trainingSet:
            #f.write(np.array2string(t.state) + t.probabilities + t.winner + t.color + os.linesep)
            #t.getAsJSON()
            #f.write(np.array2string(t.state) + str(t.winner) + str(t.color) + os.linesep)
            f.write(t.getAsJSON() + os.linesep)

def startSelfPlay(thread_count, board_size, color):
    threads = []
    for i in range(0, thread_count):
        # target = name for method that must be executed in thread
        threads.append(threading.Thread(target=startSelfPlayGame, args=('replaybuffer.txt', board_size, color)))
        threads[-1].start()
    for t in threads:
        t.join()
    # check what the heck the file had
    uniq_lines = set()
    with open('replaybuffer.txt', 'r') as f:
        for l in f:
            uniq_lines.add(l)
    for u in uniq_lines:
        sys.stdout.write(u)



#main function of all sub-programs
def main(args):
    print(args)

    if (len(sys.argv)) > 1:
        constants.configFileLocation = args[1]
    else:
        constants.configFileLocation = "../config.yaml"
    #read config file and store it in constants.py
    configFile.readConfigFile(constants.configFileLocation)

    startSelfPlay(constants.thread_count, constants.board_size, BLACK)




if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv)