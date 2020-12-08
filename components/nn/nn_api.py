import pandas as pd

from nn_model import NeuralNetwork

"""
features in unserem fall: board state
targets: move probs; value (für winner) also -1, 1

feature_names = column_names[:-1]
label_name = column_names[-1]"""

data = pd.read_json('replaybuffer.json', lines=True)

"""
momentan noch keine historie in benutzung
"""

#evtl format ändern für move, je nachdem ob cross-entropy mit jetzigem format klar kommt
#my guess: prolly not, sollte wsl einfach flat array sein, sodass index der move zahl entspricht
MOVE = data['probabilities']
#evtl flatten
WINNER = data['winner']
FEATURES = data.drop(['winner', 'probabilities'], axis=1)

print("move: {}, probs: {}, feats: {}".format(MOVE, WINNER, FEATURES))

model = NeuralNetwork()
model.summary()

#model = nn_model.create_model()
#nn_model.train_model(model)