import numpy
import numpy as np

from components.go.coords import from_flat
from components.mcts.common import TwoPlayersAbstractGameState, AbstractGameAction


class PikoAlphaGoMove(AbstractGameAction):
    def __init__(self, x_coordinate, y_coordinate, value):
        self.x_coordinate = x_coordinate
        self.y_coordinate = y_coordinate
        self.value = value

    def __repr__(self):
        return "x:{0} y:{1} v:{2}".format(
            self.x_coordinate,
            self.y_coordinate,
            self.value
        )


class GoGamestate(TwoPlayersAbstractGameState):
    WHITE = 1
    BLACK = -1

    # Position.board object from go engine
    pos = []

    def __init__(self, state, board_size, next_to_move, pos):
        if len(state.shape) != 2 or state.shape[0] != state.shape[1]:
            raise ValueError("Only 2D square boards allowed")
        self.board = state
        self.board_size = board_size
        self.next_to_move = next_to_move
        self.pos = pos  # position object by go engine

    @property
    def game_result(self):

        # Check position object

        # Game not over yet
        #TODO: Representation of double pass = 0 or None?
        if self.pos.is_game_over():
            return 0
            #return None
        elif self.pos.result() == 1:
            return 1
        elif self.pos.result() == -1:
            return -1

        return
        # return self.pos.isGameOver()

    def is_game_over(self):
        return self.pos.is_game_over()

    def move(self, move):

        if isinstance(move, int):
            move = from_flat(numpy.int64(move))
        elif np.issubdtype(move, np.integer):
            move = from_flat(move)

        if not self.pos.is_move_legal(move):
            raise ValueError(
                "move {0} on board {1} is not legal".format(move, self.pos.board)
            )

        # TODO warum wird das kopierT? --> müsste so passen (siehe original)
        # https://github.com/int8/monte-carlo-tree-search/blob/master/mctspy/games/examples/tictactoe.py
        new_pos = self.pos.play_move(move)

        return GoGamestate(new_pos.board, self.board_size, self.pos.to_play * (-1), new_pos)


    def get_legal_actions(self):
        # indices = np.where(self.board == 0)
        # return [
        #     TicTacToeMove(coords[0], coords[1], self.next_to_move)
        #     for coords in list(zip(indices[0], indices[1]))
        # ]
        # ACHTUNG: gibt array zurueck, das auch die ILLEGALEN moves enthaelt, zb. [0, 1, 0 ...]
        return self.convertPosEngineLegalMovesToOnlyLegalMovesInFlat(self.pos)

    def convertPosEngineLegalMovesToOnlyLegalMovesInFlat(self, pos):
        legalMoves = []
        counter = 0
        for move in pos.all_legal_moves(): # go engine
            if move == 1:
                legalMoves.append(counter)
            counter += 1
        return legalMoves

# wird vermutlich nicht gebraucht, da es nur die moves deklariert
# class TicTacToeMove(AbstractGameAction):
#     def __init__(self, x_coordinate, y_coordinate, value):
#         self.x_coordinate = x_coordinate
#         self.y_coordinate = y_coordinate
#         self.value = value
#
#     def __repr__(self):
#         return "x:{0} y:{1} v:{2}".format(
#             self.x_coordinate,
#             self.y_coordinate,
#             self.value
#         )
