import argparse
import yaml
import torch
import numpy as np
from collections import defaultdict, OrderedDict

from core.model_handler import ModelHandler

################################################################################
# Main #
################################################################################


# 设置种子
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


# 定义主函数
# 使用ModelHandler加载模型
def main(config):
    print_config(config)
    set_random_seed(config['random_seed'])
    model = ModelHandler(config)
    model.train()
    model.test()


# 定义多运行主函数
# 一次运行多个数据集模型
def multi_run_main(config):
    print_config(config)
    set_random_seed(config['random_seed'])
    hyperparams = []
    for k, v in config.items():
        if isinstance(v, list):
            hyperparams.append(k)

    scores = []
    # 构建多个config网格
    configs = grid(config)
    for cnf in configs:
        print('\n')
        for k in hyperparams:
            cnf['out_dir'] += '_{}_{}'.format(k, cnf[k])
        print(cnf['out_dir'])
        model = ModelHandler(cnf)
        dev_metrics = model.train()
        test_metrics = model.test()
        scores.append(test_metrics[model.model.metric_name])

    print('Average score: {}'.format(np.mean(scores)))
    print('Std score: {}'.format(np.std(scores)))


################################################################################
# ArgParse and Helper Functions #
# 参数解析和辅助函数
################################################################################
# 加载模型参数
def get_config(config_path="config.yml"):
    with open(config_path, "r") as setting:
        config1 = yaml.safe_load(setting)
    return config1


# 运行时从命令行中加载两个参数
# config：config文件的地址
# multi_run：是否为多运行
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', '--config', required=True, type=str, help='path to the config file')
    parser.add_argument('--multi_run', action='store_true', help='flag: multi run')
    args = vars(parser.parse_args())
    return args


# 打印配置参数
def print_config(config):
    print("**************** MODEL CONFIGURATION  模型配置 ****************")
    for key in sorted(config.keys()):
        val = config[key]
        keystr = "{}".format(key) + (" " * (24 - len(key)))
        print("{} -->   {}".format(keystr, val))
    print("**************** MODEL CONFIGURATION  模型配置  ****************")


# 用给定的关键字参数为这个Config类构建一个网格。用于一次性运行多个模型。
def grid(kwargs):
    """Builds a mesh grid with given keyword arguments for this Config class.
    If the value is not a list, then it is considered fixed"""

    class MncDc:
        """This is because np.meshgrid does not always work properly..."""

        def __init__(self, a):
            self.a = a  # tuple!

        def __call__(self):
            return self.a

    def merge_dicts(*dicts):
        """
        Merges dictionaries recursively. Accepts also `None` and returns always a (possibly empty) dictionary
        """
        from functools import reduce

        def merge_two_dicts(x, y):
            z = x.copy()  # start with x's keys and values
            z.update(y)  # modifies z with y's keys and values & returns None
            return z

        return reduce(lambda a, nd: merge_two_dicts(a, nd if nd else {}), dicts, {})

    sin = OrderedDict({k: v for k, v in kwargs.items() if isinstance(v, list)})
    for k, v in sin.items():
        copy_v = []
        for e in v:
            copy_v.append(MncDc(e) if isinstance(e, tuple) else e)
        sin[k] = copy_v

    grd = np.array(np.meshgrid(*sin.values()), dtype=object).T.reshape(-1, len(sin.values()))
    return [merge_dicts(
        {k: v for k, v in kwargs.items() if not isinstance(v, list)},
        {k: vv[i]() if isinstance(vv[i], MncDc) else vv[i] for i, k in enumerate(sin)}
    ) for vv in grd]


################################################################################
# Module Command-line Behavior #
# 程序入口 #
################################################################################
if __name__ == '__main__':
    cfg = get_args()
    config = get_config(cfg['config'])
    if cfg['multi_run']:
        multi_run_main(config)
    else:
        main(config)
