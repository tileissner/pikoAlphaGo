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
            constants.mcts_simulations = param_list['mcts_simulations']
            constants.amount_evaluator_iterations = param_list['amount_evaluator_iterations']
            constants.state_history_length = param_list['state_history_length']
            constants.pipeline_runs = param_list['pipeline_runs']
            constants.temperature = param_list['temperature']
            constants.competitive = param_list['competitive']
        except yaml.YAMLError as exc:
            print(exc)
