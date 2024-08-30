import argparse
import yaml
import torch
import numpy as np
from collections import defaultdict, OrderedDict

from core.model_handler import ModelHandler

################################################################################
# Main #
################################################################################



def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)



def main(config):
    print_config(config)
    set_random_seed(config['random_seed'])
    model = ModelHandler(config)
    model.train()
    model.test()




################################################################################
# ArgParse and Helper Functions #
################################################################################
# 加载模型参数
def get_config(config_path="config.yml"):
    with open(config_path, "r") as setting:
        config1 = yaml.safe_load(setting)
    return config1



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', '--config', required=True, type=str, help='path to the config file')
    parser.add_argument('--multi_run', action='store_true', help='flag: multi run')
    args = vars(parser.parse_args())
    return args


def print_config(config):
    print("**************** MODEL CONFIGURATION ****************")
    for key in sorted(config.keys()):
        val = config[key]
        keystr = "{}".format(key) + (" " * (24 - len(key)))
        print("{} -->   {}".format(keystr, val))
    print("**************** MODEL CONFIGURATION ****************")





################################################################################
# Module Command-line Behavior #
################################################################################
if __name__ == '__main__':
    cfg = get_args()
    config = get_config(cfg['config'])
    main(config)
