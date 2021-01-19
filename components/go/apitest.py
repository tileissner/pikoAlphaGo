import components.go.goEngineApi as goApi
import yaml

WHITE, NONE, BLACK = range(-1, 2)


with open("../../config.yaml", 'r') as stream:
    try:
        param_list = yaml.safe_load(stream)
        board_size = param_list['board_size'];
    except yaml.YAMLError as exc:
        print(exc)


#pos = goApi.createGame(board_size, BLACK)
#goApi.startGame(pos, BLACK)

goApi.selfPlay(board_size, BLACK)