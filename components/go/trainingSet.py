class TrainingSet:

    def __init__(self, state, probabilities, color):
        self.state = state
        self.probabilities = probabilities
        self.winner = None
        self.color = color

    def updateWinner(self, winner):
        self.winner = winner

    def saveTrainingSet(self):
        #TODO hier speichern
        return