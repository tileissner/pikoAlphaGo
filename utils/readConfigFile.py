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
            constants.improvement_percentage_threshold = param_list['improvement_percentage_threshold']
            constants.custom_batch_size = param_list['custom_batch_size']
            constants.epochs = param_list['epochs']
            constants.temperature = param_list['temperature']
            constants.competitive = param_list['competitive']
            constants.input_states = param_list['input_states']
            constants.games_per_thread = param_list['games_per_thread']
            constants.games_per_eval_thread = param_list['games_per_eval_thread']
            constants.c_puct = param_list['c_puct']

            constants.input_stack_size = constants.input_states * 2 + 1


        except yaml.YAMLError as exc:
            print(exc)
