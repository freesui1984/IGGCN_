import os
import json
import sys
from . import constants as Constants


class DummyLogger(object):
    def __init__(self, config, dirname=None, pretrained=None):
        self.config = config
        if dirname is None:
            if pretrained is None:
                raise Exception('Either --dir or --pretrained needs to be specified.')
            self.dirname = pretrained
        else:
            self.dirname = self.check_and_create_directory(dirname)
            os.makedirs(self.dirname)
            os.mkdir(os.path.join(self.dirname, 'metrics'))
            self.log_json(config, os.path.join(self.dirname, Constants._CONFIG_FILE))
        if config['logging']:
            self.f_metric = open(os.path.join(self.dirname, 'metrics', 'metrics.log'), 'a')

    def check_and_create_directory(self, dirname, prefix='train_'):
        if os.path.exists(dirname):
            existing_dirs = [d for d in os.listdir(dirname) if os.path.isdir(os.path.join(dirname, d)) and d.startswith(prefix)]

            if existing_dirs:
                existing_numbers = [int(d[len(prefix):]) for d in existing_dirs]
                next_number = max(existing_numbers) + 1
                new_dir = os.path.join(dirname, f'{prefix}{next_number}')
            else:
                new_dir = os.path.join(dirname, f'{prefix}0')

        else:
            new_dir = os.path.join(dirname, f'{prefix}0')

        return new_dir

    def log_json(self, data, filename, mode='w'):
        with open(filename, mode) as outfile:
            outfile.write(json.dumps(data, indent=4, ensure_ascii=False))

    def log(self, data, filename):
        print(data)

    def write_to_file(self, text):
        if self.config['logging']:
            self.f_metric.writelines(text + '\n')
            self.f_metric.flush()

    def close(self):
        if self.config['logging']:
            self.f_metric.close()


class Logger(object):
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass
