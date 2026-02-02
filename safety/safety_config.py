import os
import json

class SafetyConfig:
    """ base safety evaluation configurations """
    # define data members here

    def __init__(self, config_path=''):
        self.config_path = config_path # path to safety config file
        args_file = open(os.path.join(self.config_path), 'r')
        self.args = json.load(args_file)
        if self.args.get('agent_config', None) == None:
            raise ValueError
        args_file.close()
