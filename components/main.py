import yaml
from components.go.game import GameUI

def main(config):
    game = GameUI(config)
    print("hallo")
    #game.play()
    game.play_test();

if __name__ == '__main__':

    config = None
    with open('config.yaml', 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f'Error: {e}')

    if config is not None:
        main(config)
