from utils import constants
import yaml

def readConfigFile(location):
    #with open("../config.yaml", 'r') as stream:
    with open(location, 'r') as stream:
        try:
            param_list = yaml.safe_load(stream)
            constants.board_size = param_list['board_size']
            constants.location_replay_buffer = param_list['location_replay_buffer']
            constants.thread_count = param_list['thread_count']
        except yaml.YAMLError as exc:
            print(exc)
