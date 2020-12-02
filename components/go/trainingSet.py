import json
class TrainingSet:

    def __init__(self, state, probabilities, color):
        self.state = state
        self.probabilities = probabilities
        self.winner = None
        self.color = color

    def updateWinner(self, winner):
        self.winner = winner

    def getWinner(self):
        return self.winner

    def saveTrainingSet(self):
        #TODO hier speichern
        return

    #TODO: Object gets altered in the process. State is saved as list
    # not as np Array anymore. Solution: Inner class or deepcopy?
    def getAsJSON(self):
        self.state = self.state.tolist()
        return json.dumps(self.__dict__)