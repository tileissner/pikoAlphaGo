from nn_api import NetworkAPI


class Player:

    def __init__(self, color, netFileName):
        self.color = color
        self.net_api = NetworkAPI()
        self.net = self.net_api.model_load(netFileName)

