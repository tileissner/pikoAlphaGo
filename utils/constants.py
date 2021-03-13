#value will be overwritten by readConfigFile.py
configFileLocation = ''

board_size = None
location_replay_buffer = None
thread_count = None
games_per_thread = None
games_per_eval_thread = None

Q = None
mcts_simulations = None
amount_evaluator_iterations = None
state_history_length = None
pipeline_runs = None

improvement_percentage_threshold = None
input_stack_size = None
custom_batch_size = None
c_puct = None
epochs = None
temperature = None

input_states = None

currentBestNetFileName = "models/modeluntrained"
#currentBestNetFileName = "models/model"
challengerNetFileName = ""

path_to_model_to_load = None
use_old_replay_buffer = None

challenger_wins = 0
current_player_wins = 0
draws = 0