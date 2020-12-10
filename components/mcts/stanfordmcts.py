from math import sqrt
from random import random

from components.go.coords import from_flat
from components.go import goEngineApi
from components.mcts.stateProbabilities import StateProbabilities

visited = []
P = []
#P = {}
Q = []
N = []
c_puct = 0.5

def randomWinner():
    return 1 if random() < 0.5 else -1

def search(state, pos, nnet):
    #state = pos.board
    if countLegalMoves(pos) == 0:
        return -pos.result()

    if state not in visited:
        #TODO checken ob state als objekt gespeihcert wird (bzw state in visited? kommt es hier auf objekt oder auf die
        #matrix an?
        visited.append(state)
        # TODO selbiges problem? P[Objekt] oder P[x][y], wie verbucht er das intern?
        #P[state], v = nnet.predict(state) # returns tuple once implemented
        #P[state] = goEngineApi.getMockProbabilities(pos)
        #P[state] = goEngineApi.getMockProbabilities(pos)
        P.append(StateProbabilities(state, goEngineApi.getMockProbabilities(pos)))

        v = randomWinner()
        return -v

    max_u, best_a = -float("inf"), -1
    for a in convertPosEngineLegalMovesToOnlyLegalMovesInFlat(pos):
        stateProbabilities = None
        for sP in P:
            if sP.state.all() == state.all():
                stateProbabilities = sP
        #u = Q[state][a] + c_puct * stateProbabilities.probabilities[a] * sqrt(sum(N[state])) / (1 + N[state][a])
        print(type(pos))
        print(N[pos])
        u = c_puct * stateProbabilities.probabilities[a] * sqrt(sum(N[state])) / (1 + N[state][a])


        #u = Q[state][a] + c_puct * P[state][a] * sqrt(sum(N[state])) / (1 + N[state][a])
        if u > max_u:
            max_u = u
            best_a = a
    a = best_a

    successorPos = pos.play_move(from_flat(a))
    v = search(successorPos, pos, nnet)

    Q[state][a] = (N[state][a] * Q[state][a] + v) / (N[state][a] + 1)
    N[state][a] += 1
    return -v


def countLegalMoves(pos):
    counter = 0
    for move in pos.all_legal_moves(): # [0, 1, 0, 1]
        if (move == 1):
            counter += 1
    return counter

def convertPosEngineLegalMovesToOnlyLegalMovesInFlat(pos):
    'konvertiert in schöne darstellung, d.h. zb move 4, move 8 etc'
    legalMoves = []
    counter = 0
    for move in pos.all_legal_moves(): # go engine
        if move == 1:
            legalMoves.append(counter)
        counter += 1
    return legalMoves