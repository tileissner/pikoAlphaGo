# the actual pipeline.py

from Pipeline import pipelineConfig




def main_loop():

    #Train for fixed number of iterations
    for num_iteration in range(pipelineConfig.NUMBER_ITERATIONS()):
        training_samples = []

        # Play fixed number of games per iteration
        for num_episode in range(pipelineConfig.NUMBER_OF_GAMES_PER_EPISODE()):


            print('')


    return 0



# Generates Training Samples
# Takes a nnet and plays 1 game against itself
# Returns all states played during the game

# Note: Goal is to execute games in parallel
def play_against_yourself(nnet):

    #Game History
    game_states = []

    # Init game

    # Play until game is over

    return 0


#
def evaluate(current_net, new_net):


    return 0