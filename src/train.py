import sys
import json
from models import SRL

def main():
    if (len(sys.argv) == 1):
        print('Attach which configurations you want to use for the model!') 
        print('Configurations name not found.. Using the default config..')
        config = 'default'
    else :
        config = sys.argv[1]

    filename = './configurations.json'
    f = open(filename)
    all_config = json.load(f)
    
    try:
        config = all_config[config]
    except:
        print('Configurations name not found.. Using the default config..')
        config = all_config['default']

    max_tokens = config['max_tokens']
    max_char = config['max_char']
    model = SRL(config)
    model.model().summary()

if __name__ == "__main__":
    main()