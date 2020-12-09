from math import sqrt

visited = []
P = []
Q = []
N = []
c_puct = 0.5


def search(self, state, pos, nnet):
    if countLegalMoves(pos) == 0:
        return -pos.result()

    if state not in visited:
        visited.append(state)
        P[state], v = nnet.predict(state)
        return -v

    max_u, best_a = -float("inf"), -1
    for a in pos.getValidActions(state):
        u = Q[state][a] + c_puct * P[state][a] * sqrt(sum(N[state])) / (1 + N[state][a])
        if u > max_u:
            max_u = u
            best_a = a
    a = best_a

    sp = pos.play_move(state, a)
    v = search(sp, pos, nnet)

    Q[state][a] = (N[state][a] * Q[state][a] + v) / (N[state][a] + 1)
    N[state][a] += 1
    return -v


def countLegalMoves(pos):
    counter = 0
    for move in pos.all_legal_moves():
        if (move == 1):
            counter += 1
    return counter
