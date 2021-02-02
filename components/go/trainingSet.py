import json
import numpy as np
from split_input import split_input


class TrainingSet:

    def __init__(self, state, probabilities, color):
        #hier ein array von x letzten states (in listenform)
        self.state = state
        #self.probabilities = self.createProbability2DArray(probabilities)
        self.probabilities = probabilities
        self.winner = None
        self.color = color

    def updateWinner(self, winner):
        self.winner = winner

    def getWinner(self):
        return self.winner


    #TODO: Object gets altered in the process. State is saved as list
    # not as np Array anymore. Solution: Inner class or deepcopy?
    def getAsJSON(self, lastElement):
        self.state = self.state.tolist()
        if not lastElement:
            return json.dumps(self.__dict__) + ","
        else:
            return json.dumps(self.__dict__)
        #return json.dumps(self.__dict__)

    def getAsJSONWithPreviousStates(self, lastElement, previousStates):
        currentState = self.state

        backupState = self.state
        self.state = []

        # den aktuellen zustand als letztes anhaengen als 0tes element
        self.state.append(currentState.tolist())

        for previousState in previousStates:
            self.state.append(previousState.tolist())

        self.state = np.array(self.state)
        self.state = split_input(self.state)
        self.state = self.state.tolist()

        #TODO warum im 1. durchlauf array und danach liste?
        if not isinstance(self.probabilities, list):
            self.probabilities = self.probabilities.tolist()

        if not lastElement:
            returnValue = json.dumps(self.__dict__) + ","
        else:
            returnValue = json.dumps(self.__dict__)

        self.state = backupState
        return returnValue

    def getBestActionFromProbabilities(self):
        maxValue = 0
        maxKey = 0
        for key, value in self.probabilities.items():
            if (value > maxValue):
                maxValue = value
                maxKey = key
        return maxValue

    def createProbability2DArray(self, probabilities):
        twoDimensionalProbabilities = []
        for i in range(0, len(probabilities)-1, 5):
            twoDimensionalProbabilities.append(probabilities[i:i+5])

        #print(twoDimensionalProbabilities)
        return twoDimensionalProbabilities
