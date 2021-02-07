from components.nn.nn_api import NetworkAPI


class Player:

    def __init__(self, color, netFileName):
        self.color = color
        self.net_api = NetworkAPI()
        self.net_api.model_load(netFileName)
        self.mcts = None
