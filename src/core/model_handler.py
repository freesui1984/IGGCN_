import os
import time
import json
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, auc, accuracy_score, precision_score, recall_score, f1_score, \
    roc_auc_score, roc_curve, confusion_matrix
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from .model import Model
from .utils.generic_utils import to_cuda
from .utils.data_utils import prepare_datasets
from .utils.data_utils import DataStream, EEGDataStream
from .utils.data_utils import vectorize_input_eeg, vectorize_input_text
from .utils import Timer, DummyLogger, AverageMeter
from .utils import constants as Constants
from .layers.common import dropout
from .layers.anchor import sample_anchors, batch_sample_anchors, batch_select_from_tensor, compute_anchor_adj
import warnings
warnings.filterwarnings("ignore")


class ModelHandler(object):
    """High level model_handler that trains/validates/tests the network,
    tracks and logs metrics.
    """
    def __init__(self, config):
        # Evaluation Metrics:
        # 训练集和验证集
        self._train_loss = AverageMeter()
        self._dev_loss = AverageMeter()
        self.data_type = config['data_type']
        if config['dataset_name'] == 'TUEP_EEG':
            self.dataset_index = pd.read_csv(config['index_file_path'], dtype={"patient_ID": str})
        if config['dataset_name'] == 'TUAB_EEG':
            self.dataset_index = pd.read_csv(config['index_file_path'], dtype={"patient_ID": str}, index_col=0)
        if config['dataset_name'] == 'TUSZ_EEG':
            self.dataset_index = pd.read_csv(config['index_file_path'], dtype={"patient_ID": str}, index_col=0)
        # if config['dataset_name'] == 'TUSZ_EEG':
        #     self.dataset_train_index = pd.read_csv(config['train_index_file_path'], dtype={"patient_ID": str}, index_col=0)
        #     self.dataset_dev_index = pd.read_csv(config['dev_index_file_path'], dtype={"patient_ID": str}, index_col=0)
        #     self.dataset_test_index = pd.read_csv(config['test_index_file_path'], dtype={"patient_ID": str}, index_col=0)

        # 分类
        if config['task_type'] == 'classification':
            self._train_metrics = {'nloss': AverageMeter(),
                                   'acc': AverageMeter(),
                                   'precision': AverageMeter(),
                                   'recall': AverageMeter(),
                                   'f1': AverageMeter(),
                                   'Weight_F1': AverageMeter()}
                                   # 'auc': AverageMeter()}
            self._dev_metrics = {'nloss': AverageMeter(),
                                 'acc': AverageMeter(),
                                 'precision': AverageMeter(),
                                 'recall': AverageMeter(),
                                 'f1': AverageMeter(),
                                 'Weight_F1': AverageMeter()}
                                 # 'auc': AverageMeter()}
            self._dev_metrics_patient = {'acc_patient': None,
                                         'precision_patient': None,
                                         'recall_patient': None,
                                         'f1_patient': None}
            self._test_metrics_patient = {'acc_patient': None,
                                          'precision_patient': None,
                                          'recall_patient': None,
                                          'f1_patient': None}
        # 回归
        elif config['task_type'] == 'regression':
            self._train_metrics = {'nloss': AverageMeter(),
                                   'r2': AverageMeter()}
            self._dev_metrics = {'nloss': AverageMeter(),
                                 'r2': AverageMeter()}
        # 异常
        else:
            # 抛出异常
            raise ValueError('Unknown task_type: {}'.format(config['task_type']))
        # ？？？？？？？？？
        self.logger = DummyLogger(config, dirname=config['out_dir'], pretrained=config['pretrained'])
        self.dirname = self.logger.dirname
        # 显卡是否可用
        if not config['no_cuda'] and torch.cuda.is_available():
            print('[ Using CUDA ]')
            self.device = torch.device('cuda' if config['cuda_id'] < 0 else 'cuda:%d' % config['cuda_id'])
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')
        config['device'] = self.device

        # 设置种子
        seed = config.get('seed', 42)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.device:
            torch.cuda.manual_seed(seed)

        # 加载数据的过程
        datasets = prepare_datasets(config)

        # Prepare datasets
        # ------------准备数据集---------------
        # network:Cora、Citeseer、Pubmed、ogbn-arxiv
        # uci: Wine、Cancer、Digits
        if config['data_type'] in ('network', 'uci'):
            config['num_feat'] = datasets['features'].shape[-1]
            config['num_class'] = datasets['labels'].max().item() + 1

            # Initialize the model
            # 初始化模型以及网络
            self.model = Model(config, train_set=datasets.get('train', None))
            self.model.network = self.model.network.to(self.device)

            # _scalable:使用锚点集
            self._n_test_examples = datasets['idx_test'].shape[0]
            self.run_epoch = self._scalable_run_whole_epoch if config.get('scalable_run', False) else self._run_whole_epoch

            self.train_loader = datasets
            self.dev_loader = datasets
            self.test_loader = datasets

        # 文本数据（MRD、20News）
        elif config['data_type'] == 'text':
            train_set = datasets['train']
            dev_set = datasets['dev']
            test_set = datasets['test']

            config['num_class'] = max([x[-1] for x in train_set + dev_set + test_set]) + 1

            self.run_epoch = self._run_batch_epoch

            # Initialize the model
            # 初始化模型和网络结构
            self.model = Model(config, train_set=datasets.get('train', None))
            self.model.network = self.model.network.to(self.device)

            self._n_train_examples = 0
            if train_set:
                self.train_loader = DataStream(train_set, self.model.vocab_model.word_vocab, config=config, isShuffle=True, isLoop=True, isSort=False)
                self._n_train_batches = self.train_loader.get_num_batch()
            else:
                self.train_loader = None

            if dev_set:
                self.dev_loader = DataStream(dev_set, self.model.vocab_model.word_vocab, config=config, isShuffle=False, isLoop=True, isSort=False)
                self._n_dev_batches = self.dev_loader.get_num_batch()
            else:
                self.dev_loader = None

            if test_set:
                self.test_loader = DataStream(test_set, self.model.vocab_model.word_vocab, config=config, isShuffle=False, isLoop=False, isSort=False, batch_size=config['batch_size'])
                self._n_test_batches = self.test_loader.get_num_batch()
                self._n_test_examples = len(test_set)
            else:
                self.test_loader = None
        # eeg数据集-----------
        else:
            train_set = datasets['train']
            dev_set = datasets['dev']
            test_set = datasets['test']

            # 于永强注释，感觉没有用
            # config['num_class'] = max([x[-1] for x in train_set + dev_set + test_set]) + 1

            self.run_epoch = self._run_batch_epoch

            # Initialize the model
            # 初始化模型和网络结构
            self.model = Model(config, train_set=datasets.get('train', None))
            self.model.network = self.model.network.to(self.device)

            self._n_train_examples = 0
            if train_set:
                self.train_loader = EEGDataStream(train_set, config=config, isShuffle=True, isLoop=True, isSort=False)
                self._n_train_batches = self.train_loader.get_num_batch()
            else:
                self.train_loader = None

            if dev_set:
                self.dev_loader = EEGDataStream(dev_set, config=config, isShuffle=True, isLoop=True, isSort=False)
                self._n_dev_batches = self.dev_loader.get_num_batch()
            else:
                self.dev_loader = None

            if test_set:
                self.test_loader = EEGDataStream(test_set, config=config, isShuffle=True, isLoop=False, isSort=False, batch_size=config['batch_size'])
                self._n_test_batches = self.test_loader.get_num_batch()
                self._n_test_examples = len(test_set)
            else:
                self.test_loader = None

        self.config = self.model.config
        self.is_test = False

    def train(self):
        if self.train_loader is None or self.dev_loader is None:
            print("No training set or dev set specified -- skipped training.")
            return

        self.is_test = False
        # 自建名为Train的计时器
        timer = Timer("Train")
        self._epoch = self._best_epoch = 0
        # 用于保存最好的指标{nloss、acc}
        self._best_metrics = {}
        # 验证集指标设置为负无穷
        for k in self._dev_metrics:
            self._best_metrics[k] = -float('inf')
        # 先重置训练集和验证集的参数
        self._reset_metrics()

        # 最大epoch：10000  或者  超过最好的epoch再过10轮
        while self._stop_condition(self._epoch, self.config['patience']):
            self._epoch += 1

            # Train phase    ***********　　　训练阶段　　*********
            # 每 print_every_epochs=1 打印一次训练进度
            if self._epoch % self.config['print_every_epochs'] == 0:
                format_str = "\n>>> Train Epoch: [{} / {}]".format(self._epoch, self.config['max_epochs'])
                print(format_str)
                self.logger.write_to_file(format_str)

            # ************  训练过程  *************
            # _scalable_run_whole_epoch  或者  _run_whole_epoch
            # 图分类是_run_batch_epoch
            self.run_epoch(self.train_loader, training=True, test=False, verbose=self.config['verbose'])

            # 每 print_every_epochs=1 打印一次训练结果
            if self._epoch % self.config['print_every_epochs'] == 0:
                format_str = "Training Epoch {} -- Loss: {:0.5f}".format(self._epoch, self._train_loss.mean())
                format_str += self.metric_to_str(self._train_metrics)
                train_epoch_time_msg = timer.interval("Training Epoch {}".format(self._epoch))
                self.logger.write_to_file(train_epoch_time_msg + '\n' + format_str)
                print(format_str)
                format_str = "\n>>> Validation Epoch: [{} / {}]".format(self._epoch, self.config['max_epochs'])
                print(format_str)
                self.logger.write_to_file(format_str)

            # Validation phase    ***********　　　验证阶段　　*********
            dev_output, dev_gold, dev_idx, score_auc = self.run_epoch(self.dev_loader, training=False, test=False, verbose=self.config['verbose']
                                                           , out_predictions=self.config['out_predictions'])
            if self.config['out_predictions']:
                dev_output = torch.stack(dev_output, 0)
                dev_gold = torch.stack(dev_gold, 0)
                # 计算acc
                dev_metric_score = score_auc
            else:
                dev_metric_score = None

            if self.config['is_patient']:
                dev_idx = torch.stack(dev_idx, 0)
                auroc_patient, precision_patient, recall_patient, f1_patient, acc_patient = collect_metrics_new(
                    dev_output, dev_gold, dev_idx, 0, stats_val_data, "dev",
                    self.dataset_index, self.config['save_y_draw_path'])
                self._dev_metrics_patient['auc_patient'] = auroc_patient
                self._dev_metrics_patient['precision_patient'] = precision_patient
                self._dev_metrics_patient['recall_patient'] = recall_patient
                self._dev_metrics_patient['f1_patient'] = f1_patient
                self._dev_metrics_patient['acc_patient'] = acc_patient

            # 每 print_every_epochs=1 打印一次验证结果
            if self._epoch % self.config['print_every_epochs'] == 0:
                format_str = "Validation Epoch {} -- Loss: {:0.5f}".format(self._epoch, self._dev_loss.mean())
                format_str += self.metric_to_str(self._dev_metrics)
                if self.config['is_patient']:
                    format_str += "\n--- patient ---"
                    format_str += self.metric_to_str(self._dev_metrics_patient)
                    format_str += "\n"
                dev_epoch_time_msg = timer.interval("Validation Epoch {}".format(self._epoch))
                self.logger.write_to_file(dev_epoch_time_msg + '\n' + format_str)
                print(format_str)

            # 进不去
            if not self.config['data_type'] in ('network', 'uci', 'text', 'eeg'):
                self.model.scheduler.step(self._dev_metrics[self.config['eary_stop_metric']].mean())
            # if self.config['eary_stop_metric'] == self.model.metric_name and dev_metric_score is not None:
            #     cur_dev_score = dev_metric_score
            # else:
            cur_dev_score = self._dev_metrics[self.config['eary_stop_metric']]

            # if dev_metric_score is not None:
            #     self._best_metrics[self.model.metric_name] = dev_metric_score

            # if self._best_metrics[self.config['eary_stop_metric']] < self._dev_metrics[self.config[
            # 'eary_stop_metric']].mean(): 找到一个新的最好的验证集效果
            if self._best_metrics[self.config['eary_stop_metric']] < cur_dev_score:
                self._best_epoch = self._epoch
                for k in self._dev_metrics:
                    self._best_metrics[k] = self._dev_metrics[k]

                if self.config['save_params']:
                    self.model.save(self.dirname)
                    if self._epoch % self.config['print_every_epochs'] == 0:
                        format_str = 'Saved model to {}'.format(self.dirname)
                        self.logger.write_to_file(format_str)
                        print(format_str)

                if self._epoch % self.config['print_every_epochs'] == 0:
                    format_str = "!!! Updated: " + self.best_metric_to_str(self._best_metrics)
                    self.logger.write_to_file(format_str)
                    print(format_str)

            # 每一轮训练结束，都要重置训练集和验证集的指标，，注意：：重置不是归零
            self._reset_metrics()

        timer.finish()

        format_str = "Finished Training: {}\nTraining time: {}".format(self.dirname, timer.total) + '\n' + self.summary()
        print(format_str)
        self.logger.write_to_file(format_str)
        return self._best_metrics

    # 测试
    def test(self):
        if self.test_loader is None:
            print("No testing set specified -- skipped testing.")
            return

        # Restore best model
        # 加载最好的模型
        print('Restoring best model')
        # 加载已经保存的模型
        self.model.init_saved_network(self.dirname)
        self.model.network = self.model.network.to(self.device)

        self.is_test = True
        self._reset_metrics()
        timer = Timer("Test")

        # 特征层中参数都固定住，不会发生梯度的更新
        # 减少计算量，也就是不用求w和b的导数了，减少了计算量。只传播误差，而不计算权重和偏执的导数。
        for param in self.model.network.parameters():
            param.requires_grad = False

        output, gold, idx, score_acc = self.run_epoch(self.test_loader, training=False, test=True, verbose=0,
                                                              out_predictions=self.config['out_predictions'],
                                                              out_adj=self.config['out_adj'])

        metrics = self._dev_metrics
        format_str = "[test] | test_exs = {} | step: [{} / {}]".format(self._n_test_examples, 1, 1)
        format_str += self.metric_to_str(metrics)

        if self.config['out_predictions']:
            output = torch.stack(output, 0)
            gold = torch.stack(gold, 0)
            # 计算auc
            test_score = score_acc
            # format_str += '\nFinal AUC score on the testing set: {:0.5f}\n'.format(test_score)
        else:
            test_score = None

        if self.config['is_patient']:
            idx = torch.stack(idx, 0)
            auroc_patient, precision_patient, recall_patient, f1_patient, acc_patient = collect_metrics_new(
                output, gold, idx, 0, stats_val_data, "test", self.dataset_index, self.config['save_y_draw_path'])
            self._test_metrics_patient['auc_patient'] = auroc_patient
            self._test_metrics_patient['precision_patient'] = precision_patient
            self._test_metrics_patient['recall_patient'] = recall_patient
            self._test_metrics_patient['f1_patient'] = f1_patient
            self._test_metrics_patient['acc_patient'] = acc_patient
            format_str += '\n--- patient ---'
            format_str += self.metric_to_str(self._test_metrics_patient)
            format_str += '\n'

        # 打印结果
        print(format_str)
        self.logger.write_to_file(format_str)
        timer.finish()

        format_str = "Finished Testing: {}\nTesting time: {}".format(self.dirname, timer.total)
        print(format_str)
        self.logger.write_to_file(format_str)
        self.logger.close()

        test_metrics = {}
        for k in metrics:
            test_metrics[k] = metrics[k]

        if test_score is not None:
            test_metrics[self.model.metric_name] = test_score

        if self.config['out_adj']:
            save_init_adj = torch.stack(init_adj).numpy()
            save_opt_adj = torch.stack(opt_adj).numpy()
            save_adj(save_init_adj, save_opt_adj, self.config['save_adj_path'])
        return test_metrics

    # 不用图神经网络
    def batch_no_gnn(self, x_batch, step, training, out_predictions=False):
        '''Iterative graph learning: batch training'''
        mode = "train" if training else ("test" if self.is_test else "dev")
        network = self.model.network
        network.train(training)

        context, context_lens, targets = x_batch['context'], x_batch['context_lens'], x_batch['targets']
        context2 = x_batch.get('context2', None)
        context2_lens = x_batch.get('context2_lens', None)

        output = network.compute_no_gnn_output(context, context_lens)

        # BP to update weights
        loss = self.model.criterion(output, targets)
        score = self.model.score_func(targets.cpu(), output.detach().cpu())

        res = {'loss': loss.item(),
                'metrics': {'nloss': -loss.item(), self.model.metric_name: score},
        }
        if out_predictions:
            res['predictions'] = output.detach().cpu()

        if training:
            loss = loss / self.config['grad_accumulated_steps'] # Normalize our loss (if averaged)
            loss.backward()

            if (step + 1) % self.config['grad_accumulated_steps'] == 0: # Wait for several backward steps
                self.model.clip_grad()
                self.model.optimizer.step()
                self.model.optimizer.zero_grad()
        return res



    # 迭代图学习，适用EEG-----------
    def eeg_batch_IGL_stop(self, x_batch, step, training, out_predictions=False, out_adj=False):
        '''Iterative graph learning: batch training, batch stopping'''
        # 加载模型
        mode = "train" if training else ("test" if self.is_test else "dev")
        network = self.model.network
        network.train(training)

        # 获取数据集的特征，长度，标签
        context, targets, adj = x_batch['context'], x_batch['targets'], x_batch['adj']
        context_lens = x_batch['context_lens']
        seq_lengths = self.config['min_word_freq']
        seq_lengths = np.full((x_batch['batch_size'],), fill_value=seq_lengths)



        # Prepare init node embedding, init adj
        # 准备初始的节点嵌入和图拓扑
        # raw_context_vec, context_vec, context_mask, init_adj
        # [batch, 464] --->  [batch, 16, 29]
        raw_context_vec, context_vec, init_adj = network.prepare_init_graph(context, context_lens, adj)
        # 更改应用eeg

        # Init
        raw_node_vec = raw_context_vec # word embedding
        init_node_vec = context_vec # hidden embedding
        node_mask = None

        # 这条会出现warning
        if self.config['graph_learn']:
            cur_raw_adj, cur_adj = network.learn_graph(network.graph_learner, raw_node_vec, network.graph_skip_conn, node_mask=node_mask, graph_include_self=network.graph_include_self, init_adj=init_adj)

        if self.config['graph_module'] == 'gcn':
            # GCN
            node_vec = torch.relu(network.encoder.graph_encoders[0](init_node_vec, cur_adj))
            node_vec = F.dropout(node_vec, network.dropout, training=network.training)

            # Add mid GNN layers
            # 中间层
            for encoder in network.encoder.graph_encoders[1:-1]:
                node_vec = torch.relu(encoder(node_vec, cur_adj))
                node_vec = F.dropout(node_vec, network.dropout, training=network.training)

            # BP to update weights
            output = network.encoder.graph_encoders[-1](node_vec, cur_adj)
            output = network.compute_output(output, node_mask=node_mask)
        elif self.config['graph_module'] == 'ggnn' and self.config['graph_learn']:
            # GGNN
            node_vec = torch.relu(network.encoder(init_node_vec, cur_adj))
            # node_vec = F.dropout(node_vec, network.dropout, training=network.training)

            # # 中间层
            # node_vec = torch.relu(network.encoder(node_vec, cur_adj))
            # node_vec = F.dropout(node_vec, network.dropout, training=network.training)

            # # BP to update weights
            # output = network.encoder(node_vec, cur_adj)
            output = network.compute_output(node_vec, node_mask=node_mask)
        # if args.model_name == "dcrnn":
        #     logits = model(x, seq_lengths, supports)
        # elif args.model_name == "densecnn":
        #     x = x.transpose(-1, -2).reshape(batch_size, -1, args.num_nodes)  # (batch_size, seq_len, num_nodes)
        #     logits = model(x)
        # elif args.model_name == "lstm" or args.model_name == "cnnlstm":
        #     logits = model(x, seq_lengths)
        # else:
        #     raise NotImplementedError
        # if logits.shape[-1] == 1:
        #     logits = logits.view(-1)  # (batch_size,)
        elif self.config['graph_module'] == 'ggnn' and not self.config['graph_learn']:
            node_vec = torch.relu(network.encoder(init_node_vec, init_adj))
            node_vec = F.dropout(node_vec, network.dropout, training=network.training)
            output = network.compute_output(node_vec, node_mask=node_mask)
        elif self.config['graph_module'] == 'lstm':
            output = network.encoder(context, seq_lengths)
        elif self.config['graph_module'] == 'cnnlstm':
            output = network.encoder(context, seq_lengths)
        elif self.config['graph_module'] == 'densecnn':
            output = network.encoder(context)
        else:
            raise NotImplementedError
        loss1 = self.model.criterion(output, targets)

        if self.config['graph_learn'] and self.config['graph_learn_regularization']:
            graph_loss1 = self.add_batch_graph_loss(cur_raw_adj, raw_node_vec)
            loss1 += graph_loss1
        if self.config['graph_learn']:
            first_raw_adj, first_adj = cur_raw_adj, cur_adj

        if not mode == 'test':
            if self._epoch > self.config.get('pretrain_epoch', 0):
                max_iter_ = self.config.get('max_iter', 10) # Fine-tuning
                if self._epoch == self.config.get('pretrain_epoch', 0) + 1:
                    for k in self._dev_metrics:
                        self._best_metrics[k] = -float('inf')

            else:
                max_iter_ = 0 # Pretraining
        else:
            max_iter_ = self.config.get('max_iter', 10)

        #
        eps_adj = float(self.config.get('eps_adj', 0)) if training else float(self.config.get('test_eps_adj', self.config.get('eps_adj', 0)))

        loss = 0
        iter_ = 0

        # Indicate the last iteration number for each example
        batch_last_iters = to_cuda(torch.zeros(x_batch['batch_size'], dtype=torch.uint8), self.device)
        # Indicate either an example is in onging state (i.e., 1) or stopping state (i.e., 0)
        batch_stop_indicators = to_cuda(torch.ones(x_batch['batch_size'], dtype=torch.uint8), self.device)
        batch_all_outputs = []
        while self.config['graph_learn'] and (iter_ == 0 or torch.sum(batch_stop_indicators).item() > 0) and iter_ < max_iter_:
            iter_ += 1
            batch_last_iters += batch_stop_indicators
            pre_adj = cur_adj
            pre_raw_adj = cur_raw_adj
            cur_raw_adj, cur_adj = network.learn_graph(network.graph_learner2, node_vec, network.graph_skip_conn, node_mask=node_mask, graph_include_self=network.graph_include_self, init_adj=init_adj)

            update_adj_ratio = self.config.get('update_adj_ratio', None)
            if update_adj_ratio is not None:
                cur_adj = update_adj_ratio * cur_adj + (1 - update_adj_ratio) * first_adj
            if self.config['graph_module'] == 'gcn':
                # GCN
                node_vec = torch.relu(network.encoder.graph_encoders[0](init_node_vec, cur_adj))
                node_vec = F.dropout(node_vec, self.config.get('gl_dropout', 0), training=network.training)

                # Add mid GNN layers
                for encoder in network.encoder.graph_encoders[1:-1]:
                    node_vec = torch.relu(encoder(node_vec, cur_adj))
                    node_vec = F.dropout(node_vec, self.config.get('gl_dropout', 0), training=network.training)

                # BP to update weights
                tmp_output = network.encoder.graph_encoders[-1](node_vec, cur_adj)
                tmp_output = network.compute_output(tmp_output, node_mask=node_mask)
            else:
                # GGNN
                node_vec = torch.relu(network.encoder(init_node_vec, cur_adj))
                tmp_output = network.compute_output(node_vec, node_mask=node_mask)
            batch_all_outputs.append(tmp_output.unsqueeze(1))

            tmp_loss = self.model.criterion(tmp_output, targets)
            if len(tmp_loss.shape) == 2:
                tmp_loss = torch.mean(tmp_loss, 1)

            loss += batch_stop_indicators.float() * tmp_loss

            # 损失正则化
            if self.config['graph_learn'] and self.config['graph_learn_regularization']:
                graph_loss = batch_stop_indicators.float() * self.add_batch_graph_loss(cur_raw_adj, raw_node_vec, keep_batch_dim=True)
                loss += graph_loss

            if self.config['graph_learn'] and not self.config.get('graph_learn_ratio', None) in (None, 0):
                loss += batch_stop_indicators.float() * batch_SquaredFrobeniusNorm(cur_raw_adj - pre_raw_adj) * self.config.get('graph_learn_ratio')

            tmp_stop_criteria = batch_diff(cur_raw_adj, pre_raw_adj, first_raw_adj) > eps_adj
            batch_stop_indicators = batch_stop_indicators * tmp_stop_criteria

        if iter_ > 0:
            loss = torch.mean(loss / batch_last_iters.float()) + loss1
            graph_loss = torch.mean(graph_loss / batch_last_iters.float()) + graph_loss1
            # print('[# 训练图损失: {}] '.format(graph_loss))

            batch_all_outputs = torch.cat(batch_all_outputs, 1)
            selected_iter_index = batch_last_iters.long().unsqueeze(-1) - 1

            if len(batch_all_outputs.shape) == 3:
                selected_iter_index = selected_iter_index.unsqueeze(-1).expand(-1, -1, batch_all_outputs.size(-1))
                output = batch_all_outputs.gather(1, selected_iter_index).squeeze(1)
            else:
                output = batch_all_outputs.gather(1, selected_iter_index)


        else:
            loss = loss1
            # print('[# 验证图损失: {}] '.format(graph_loss1))

        if training:
            res = {'loss': loss.item(),
                   }
        else:
            res = {'loss': loss.item(),
                   }
        if out_predictions:
            res['predictions'] = output.detach().cpu()
            res['idx'] = x_batch['idx'].cpu()
        # 保存新老adj
        if self.config['graph_module'] == 'ggnn' and self.config['graph_learn']:
            res['init_adj'] = adj.detach().cpu()
            shape_0 = cur_adj.shape[0]
            shape_1 = cur_adj.shape[1]
            shape_2 = cur_adj.shape[2]
            res['opt_adj'] = cur_adj.reshape(shape_0, shape_1*shape_2).detach().cpu()
        if training:
            loss = loss / self.config['grad_accumulated_steps'] # Normalize our loss (if averaged)
            loss.backward()

            if (step + 1) % self.config['grad_accumulated_steps'] == 0: # Wait for several backward steps
                self.model.clip_grad()
                self.model.optimizer.step()
                self.model.optimizer.zero_grad()
        return res, output

    # 使用锚点集
    def scalable_batch_IGL_stop(self, x_batch, step, training, out_predictions=False):
        '''Iterative graph learning: batch training, batch stopping'''
        mode = "train" if training else ("test" if self.is_test else "dev")
        network = self.model.network
        network.train(training)

        context, context_lens, targets = x_batch['context'], x_batch['context_lens'], x_batch['targets']
        context2 = x_batch.get('context2', None)
        context2_lens = x_batch.get('context2_lens', None)

        # Prepare init node embedding, init adj
        raw_context_vec, context_vec, context_mask, init_adj = network.prepare_init_graph(context, context_lens)

        # Init
        raw_node_vec = raw_context_vec # word embedding
        init_node_vec = context_vec # hidden embedding
        node_mask = context_mask

        # Randomly sample s anchor nodes
        init_anchor_vec, anchor_mask, sampled_node_idx, max_num_anchors = batch_sample_anchors(init_node_vec, network.config.get('ratio_anchors', 0.2), node_mask=node_mask, device=self.device)
        raw_anchor_vec = batch_select_from_tensor(raw_node_vec, sampled_node_idx, max_num_anchors, self.device)

        # Compute n x s node-anchor relationship matrix
        cur_node_anchor_adj = network.learn_graph(network.graph_learner, raw_node_vec, anchor_features=raw_anchor_vec, node_mask=node_mask, anchor_mask=anchor_mask)

        # Compute s x s anchor graph
        cur_anchor_adj = compute_anchor_adj(cur_node_anchor_adj, anchor_mask=anchor_mask)

        # Update node embeddings via node-anchor-node message passing
        init_agg_vec = network.encoder.graph_encoders[0](init_node_vec, init_adj, anchor_mp=False, batch_norm=False)
        node_vec = (1 - network.graph_skip_conn) * network.encoder.graph_encoders[0](init_node_vec, cur_node_anchor_adj, anchor_mp=True, batch_norm=False) + \
                    network.graph_skip_conn * init_agg_vec

        if network.encoder.graph_encoders[0].bn is not None:
            node_vec = network.encoder.graph_encoders[0].compute_bn(node_vec)

        node_vec = torch.relu(node_vec)
        node_vec = F.dropout(node_vec, network.dropout, training=network.training)
        anchor_vec = batch_select_from_tensor(node_vec, sampled_node_idx, max_num_anchors, self.device)

        first_node_anchor_adj, first_anchor_adj = cur_node_anchor_adj, cur_anchor_adj
        first_init_agg_vec = network.encoder.graph_encoders[0](init_node_vec, first_node_anchor_adj, anchor_mp=True, batch_norm=False)

        # Add mid GNN layers
        for encoder in network.encoder.graph_encoders[1:-1]:
            node_vec = (1 - network.graph_skip_conn) * encoder(node_vec, cur_node_anchor_adj, anchor_mp=True, batch_norm=False) + \
                        network.graph_skip_conn * encoder(node_vec, init_adj, anchor_mp=False, batch_norm=False)

            if encoder.bn is not None:
                node_vec = encoder.compute_bn(node_vec)

            node_vec = torch.relu(node_vec)
            node_vec = F.dropout(node_vec, network.dropout, training=network.training)
            anchor_vec = batch_select_from_tensor(node_vec, sampled_node_idx, max_num_anchors, self.device)

        # Compute output via node-anchor-node message passing
        output = (1 - network.graph_skip_conn) * network.encoder.graph_encoders[-1](node_vec, cur_node_anchor_adj, anchor_mp=True, batch_norm=False) + \
                    network.graph_skip_conn * network.encoder.graph_encoders[-1](node_vec, init_adj, anchor_mp=False, batch_norm=False)
        output = network.compute_output(output, node_mask=node_mask)
        loss1 = self.model.criterion(output, targets)
        score = self.model.score_func(targets.cpu(), output.detach().cpu())

        if self.config['graph_learn'] and self.config['graph_learn_regularization']:
            loss1 += self.add_batch_graph_loss(cur_anchor_adj, raw_anchor_vec)

        if not mode == 'test':
            if self._epoch > self.config.get('pretrain_epoch', 0):
                max_iter_ = self.config.get('max_iter', 10) # Fine-tuning
                if self._epoch == self.config.get('pretrain_epoch', 0) + 1:
                    for k in self._dev_metrics:
                        self._best_metrics[k] = -float('inf')

            else:
                max_iter_ = 0 # Pretraining
        else:
            max_iter_ = self.config.get('max_iter', 10)

        eps_adj = float(self.config.get('eps_adj', 0)) if training else float(self.config.get('test_eps_adj', self.config.get('eps_adj', 0)))
        pre_node_anchor_adj = cur_node_anchor_adj

        loss = 0
        iter_ = 0
        # Indicate the last iteration number for each example
        batch_last_iters = to_cuda(torch.zeros(x_batch['batch_size'], dtype=torch.uint8), self.device)
        # Indicate either an example is in onging state (i.e., 1) or stopping state (i.e., 0)
        batch_stop_indicators = to_cuda(torch.ones(x_batch['batch_size'], dtype=torch.uint8), self.device)
        batch_all_outputs = []
        while self.config['graph_learn'] and (iter_ == 0 or torch.sum(batch_stop_indicators).item() > 0) and iter_ < max_iter_:
            iter_ += 1
            batch_last_iters += batch_stop_indicators
            pre_node_anchor_adj = cur_node_anchor_adj

            # Compute n x s node-anchor relationship matrix
            cur_node_anchor_adj = network.learn_graph(network.graph_learner2, node_vec, anchor_features=anchor_vec, node_mask=node_mask, anchor_mask=anchor_mask)

            # Compute s x s anchor graph
            cur_anchor_adj = compute_anchor_adj(cur_node_anchor_adj, anchor_mask=anchor_mask)

            cur_agg_vec = network.encoder.graph_encoders[0](init_node_vec, cur_node_anchor_adj, anchor_mp=True, batch_norm=False)

            update_adj_ratio = self.config.get('update_adj_ratio', None)
            if update_adj_ratio is not None:
                cur_agg_vec = update_adj_ratio * cur_agg_vec + (1 - update_adj_ratio) * first_init_agg_vec

            node_vec = (1 - network.graph_skip_conn) * cur_agg_vec + \
                    network.graph_skip_conn * init_agg_vec

            if network.encoder.graph_encoders[0].bn is not None:
                node_vec = network.encoder.graph_encoders[0].compute_bn(node_vec)

            node_vec = torch.relu(node_vec)
            node_vec = F.dropout(node_vec, self.config.get('gl_dropout', 0), training=network.training)
            anchor_vec = batch_select_from_tensor(node_vec, sampled_node_idx, max_num_anchors, self.device)

            # Add mid GNN layers
            for encoder in network.encoder.graph_encoders[1:-1]:
                mid_cur_agg_vec = encoder(node_vec, cur_node_anchor_adj, anchor_mp=True, batch_norm=False)
                if update_adj_ratio is not None:
                    mid_first_agg_vecc = encoder(node_vec, first_node_anchor_adj, anchor_mp=True, batch_norm=False)
                    mid_cur_agg_vec = update_adj_ratio * mid_cur_agg_vec + (1 - update_adj_ratio) * mid_first_agg_vecc

                node_vec = (1 - network.graph_skip_conn) * mid_cur_agg_vec + \
                        network.graph_skip_conn * encoder(node_vec, init_adj, anchor_mp=False, batch_norm=False)

                if encoder.bn is not None:
                    node_vec = encoder.compute_bn(node_vec)

                node_vec = torch.relu(node_vec)
                node_vec = F.dropout(node_vec, self.config.get('gl_dropout', 0), training=network.training)
                anchor_vec = batch_select_from_tensor(node_vec, sampled_node_idx, max_num_anchors, self.device)

            cur_agg_vec = network.encoder.graph_encoders[-1](node_vec, cur_node_anchor_adj, anchor_mp=True, batch_norm=False)
            if update_adj_ratio is not None:
                first_agg_vec = network.encoder.graph_encoders[-1](node_vec, first_node_anchor_adj, anchor_mp=True, batch_norm=False)
                cur_agg_vec = update_adj_ratio * cur_agg_vec + (1 - update_adj_ratio) * first_agg_vec

            tmp_output = (1 - network.graph_skip_conn) * cur_agg_vec + \
                    network.graph_skip_conn * network.encoder.graph_encoders[-1](node_vec, init_adj, anchor_mp=False, batch_norm=False)

            tmp_output = network.compute_output(tmp_output, node_mask=node_mask)
            batch_all_outputs.append(tmp_output.unsqueeze(1))

            tmp_loss = self.model.criterion(tmp_output, targets, reduction='none')
            if len(tmp_loss.shape) == 2:
                tmp_loss = torch.mean(tmp_loss, 1)

            loss += batch_stop_indicators.float() * tmp_loss

            if self.config['graph_learn'] and self.config['graph_learn_regularization']:
                loss += batch_stop_indicators.float() * self.add_batch_graph_loss(cur_anchor_adj, raw_anchor_vec, keep_batch_dim=True)

            if self.config['graph_learn'] and not self.config.get('graph_learn_ratio', None) in (None, 0):
                loss += batch_stop_indicators.float() * batch_SquaredFrobeniusNorm(cur_node_anchor_adj - pre_node_anchor_adj) * self.config.get('graph_learn_ratio')

            tmp_stop_criteria = batch_diff(cur_node_anchor_adj, pre_node_anchor_adj, cur_node_anchor_adj) > eps_adj
            batch_stop_indicators = batch_stop_indicators * tmp_stop_criteria

        if iter_ > 0:
            loss = torch.mean(loss / batch_last_iters.float()) + loss1

            batch_all_outputs = torch.cat(batch_all_outputs, 1)
            selected_iter_index = batch_last_iters.long().unsqueeze(-1) - 1

            if len(batch_all_outputs.shape) == 3:
                selected_iter_index = selected_iter_index.unsqueeze(-1).expand(-1, -1, batch_all_outputs.size(-1))
                output = batch_all_outputs.gather(1, selected_iter_index).squeeze(1)
            else:
                output = batch_all_outputs.gather(1, selected_iter_index)

            score = self.model.score_func(targets.cpu(), output.detach().cpu())

        else:
            loss = loss1

        res = {'loss': loss.item(),
                'metrics': {'nloss': -loss.item(), self.model.metric_name: score},
        }
        if out_predictions:
            res['predictions'] = output.detach().cpu()

        if training:
            loss = loss / self.config['grad_accumulated_steps'] # Normalize our loss (if averaged)
            loss.backward()

            if (step + 1) % self.config['grad_accumulated_steps'] == 0: # Wait for several backward steps
                self.model.clip_grad()
                self.model.optimizer.step()
                self.model.optimizer.zero_grad()
        return res

    # 节点分类的迭代过程
    # 每个epoch运行所有的数据集，不分batch
    # data_loader：prepare_datasets
    def _run_whole_epoch(self, data_loader, training=True, verbose=None, out_predictions=False):
        '''BP after all iterations'''
        mode = "train" if training else ("test" if self.is_test else "dev")
        # model：处理底层网络初始化的高级模型
        # network：底层图分类以及图结构学习的模型
        self.model.network.train(training)

        # 加载初始邻接矩阵、特征以及标签
        init_adj, features, labels = data_loader['adj'], data_loader['features'], data_loader['labels']

        if mode == 'train':
            idx = data_loader['idx_train']
        elif mode == 'dev':
            idx = data_loader['idx_val']
        else:
            idx = data_loader['idx_test']

        network = self.model.network

        # Init
        # 初始化
        features = F.dropout(features, network.config.get('feat_adj_dropout', 0), training=network.training)
        init_node_vec = features

        # 先进行一个图结构学习   去 graph_clf.py 里去看
        # raw_adj:最初图拓扑学习计算出来的邻接矩阵
        # adj:经过归一化和其他操作之后的邻接矩阵
        cur_raw_adj, cur_adj = network.learn_graph(network.graph_learner, init_node_vec, network.graph_skip_conn, graph_include_self=network.graph_include_self, init_adj=init_adj)
        if self.config['graph_learn'] and self.config.get('max_iter', 10) > 0:
            cur_raw_adj = F.dropout(cur_raw_adj, network.config.get('feat_adj_dropout', 0), training=network.training)
        cur_adj = F.dropout(cur_adj, network.config.get('feat_adj_dropout', 0), training=network.training)

        # 此时encoder是GAT()
        if network.graph_module == 'gat':
            assert self.config['graph_learn'] is False and self.config.get('max_iter', 10) == 0
            # 进入图神经网络
            node_vec = network.encoder(init_node_vec, init_adj)
            output = F.log_softmax(node_vec, dim=-1)

        # 此时encoder是GraphSAGE()
        elif network.graph_module == 'graphsage':
            assert self.config['graph_learn'] is False and self.config.get('max_iter', 10) == 0
            # Convert adj to DGLGraph
            import dgl
            from scipy import sparse
            binarized_adj = sparse.coo_matrix(init_adj.detach().cpu().numpy() != 0)
            dgl_graph = dgl.DGLGraph(binarized_adj)

            node_vec = network.encoder(dgl_graph, init_node_vec)
            output = F.log_softmax(node_vec, dim=-1)

        # 此时encoder是GCN()  去gnn.py这个文件里看
        else:
            # graph_encoders是一个储存不同 module，并自动将每个 module 的 parameters 添加到网络之中的容器
            # graph_encoders[0]是 GCNLayer(nfeat, nhid, batch_norm=batch_norm)
            node_vec = torch.relu(network.encoder.graph_encoders[0](init_node_vec, cur_adj))
            node_vec = F.dropout(node_vec, network.dropout, training=network.training)

            # Add mid GNN layers
            for encoder in network.encoder.graph_encoders[1:-1]:
                node_vec = torch.relu(encoder(node_vec, cur_adj))
                node_vec = F.dropout(node_vec, network.dropout, training=network.training)

            # BP to update weights
            output = network.encoder.graph_encoders[-1](node_vec, cur_adj)
            output = F.log_softmax(output, dim=-1)

        # 计算得分以及损失
        score = self.model.score_func(labels[idx], output[idx])
        loss1 = self.model.criterion(output[idx], labels[idx])

        #
        if self.config['graph_learn'] and self.config['graph_learn_regularization']:
            loss1 += self.add_graph_loss(cur_raw_adj, init_node_vec)

        # 第一次计算出的图拓扑保存为first_adj
        first_raw_adj, first_adj = cur_raw_adj, cur_adj

        # max_iter_：最大迭代次数
        if not mode == 'test':
            if self._epoch > self.config.get('pretrain_epoch', 0):
                max_iter_ = self.config.get('max_iter', 10) # Fine-tuning
                # 第一次训练，将最好的指标初始化
                if self._epoch == self.config.get('pretrain_epoch', 0) + 1:
                    for k in self._dev_metrics:
                        self._best_metrics[k] = -float('inf')

            else:
                max_iter_ = 0 # Pretraining
        else:
            max_iter_ = self.config.get('max_iter', 10)

        if training:
            # cora: 5.5e-8, cora w/o input graph: 1e-8, citeseer w/o input graph: 1e-8, wine: 2e-5, cancer: 2e-5, digtis: 2e-5
            eps_adj = float(self.config.get('eps_adj', 0))
        else:
            eps_adj = float(self.config.get('test_eps_adj', self.config.get('eps_adj', 0)))

        pre_raw_adj = cur_raw_adj
        pre_adj = cur_adj

        loss = 0
        iter_ = 0

        # 测试集和训练集都会进来
        while self.config['graph_learn'] and (iter_ == 0 or diff(cur_raw_adj, pre_raw_adj, first_raw_adj).item() > eps_adj) and iter_ < max_iter_:
            iter_ += 1
            pre_adj = cur_adj
            pre_raw_adj = cur_raw_adj

            # 计算本轮的当前的图拓扑-----感觉有错误init_adj不是应该是pre_adj
            cur_raw_adj, cur_adj = network.learn_graph(network.graph_learner2, node_vec, network.graph_skip_conn, graph_include_self=network.graph_include_self, init_adj=init_adj)

            # update_adj_ratio：每一次更新adj是
            update_adj_ratio = self.config.get('update_adj_ratio', None)
            if update_adj_ratio is not None:
                cur_adj = update_adj_ratio * cur_adj + (1 - update_adj_ratio) * first_adj

            # 第一层图卷积
            # 感觉这里也有点问题init_node_vec不是node_vec吗
            node_vec = torch.relu(network.encoder.graph_encoders[0](init_node_vec, cur_adj))
            node_vec = F.dropout(node_vec, self.config.get('gl_dropout', 0), training=network.training)

            # Add mid GNN layers
            for encoder in network.encoder.graph_encoders[1:-1]:
                node_vec = torch.relu(encoder(node_vec, cur_adj))
                node_vec = F.dropout(node_vec, self.config.get('gl_dropout', 0), training=network.training)

            # BP to update weights
            output = network.encoder.graph_encoders[-1](node_vec, cur_adj)
            output = F.log_softmax(output, dim=-1)
            score = self.model.score_func(labels[idx], output[idx])
            loss += self.model.criterion(output[idx], labels[idx])

            if self.config['graph_learn'] and self.config['graph_learn_regularization']:
                loss += self.add_graph_loss(cur_raw_adj, init_node_vec)

            if self.config['graph_learn'] and not self.config.get('graph_learn_ratio', None) in (None, 0):
                loss += SquaredFrobeniusNorm(cur_raw_adj - pre_raw_adj) * self.config.get('graph_learn_ratio')

        # 将测试集的cur_raw_adj保存到日志里out文件夹
        if mode == 'test' and self.config.get('out_raw_learned_adj_path', None):
            out_raw_learned_adj_path = os.path.join(self.dirname, self.config['out_raw_learned_adj_path'])
            np.save(out_raw_learned_adj_path, cur_raw_adj.cpu())
            print('Saved raw_learned_adj to {}'.format(out_raw_learned_adj_path))

        if iter_ > 0:
            loss = loss / iter_ + loss1
        else:
            loss = loss1

        # 反向传播，反向传播是在每一轮的10次迭代之后
        # 反向传播的时候model会变，network会变，下轮训练使用原始数据和更新后的model就可以完成训练
        if training:
            self.model.optimizer.zero_grad()
            loss.backward()
            self.model.clip_grad()
            self.model.optimizer.step()

        # 更新评价指标
        self._update_metrics(loss.item(), {'nloss': -loss.item(), self.model.metric_name: score}, 1, training=training)
        return output[idx], labels[idx]

    # 可伸缩的节点分类迭代过程
    def _scalable_run_whole_epoch(self, data_loader, training=True, verbose=None, out_predictions=False):
        '''Scalable run: BP after all iterations'''
        mode = "train" if training else ("test" if self.is_test else "dev")
        self.model.network.train(training)

        init_adj, features, labels = data_loader['adj'], data_loader['features'], data_loader['labels']

        if mode == 'train':
            idx = data_loader['idx_train']
        elif mode == 'dev':
            idx = data_loader['idx_val']
        else:
            idx = data_loader['idx_test']

        network = self.model.network

        # Init
        features = F.dropout(features, network.config.get('feat_adj_dropout', 0), training=network.training)
        init_node_vec = features

        # Randomly sample s anchor nodes
        init_anchor_vec, sampled_node_idx = sample_anchors(init_node_vec, network.config.get('num_anchors', int(0.2 * init_node_vec.size(0))))

        # Compute n x s node-anchor relationship matrix
        cur_node_anchor_adj = network.learn_graph(network.graph_learner, init_node_vec, anchor_features=init_anchor_vec)

        # Compute s x s anchor graph
        cur_anchor_adj = compute_anchor_adj(cur_node_anchor_adj)

        if self.config['graph_learn'] and self.config.get('max_iter', 10) > 0:
            cur_node_anchor_adj = F.dropout(cur_node_anchor_adj, network.config.get('feat_adj_dropout', 0), training=network.training)

        cur_anchor_adj = F.dropout(cur_anchor_adj, network.config.get('feat_adj_dropout', 0), training=network.training)

        # Update node embeddings via node-anchor-node message passing
        init_agg_vec = network.encoder.graph_encoders[0](init_node_vec, init_adj, anchor_mp=False, batch_norm=False)
        node_vec = (1 - network.graph_skip_conn) * network.encoder.graph_encoders[0](init_node_vec, cur_node_anchor_adj, anchor_mp=True, batch_norm=False) + \
                    network.graph_skip_conn * init_agg_vec

        if network.encoder.graph_encoders[0].bn is not None:
            node_vec = network.encoder.graph_encoders[0].compute_bn(node_vec)

        node_vec = torch.relu(node_vec)
        node_vec = F.dropout(node_vec, network.dropout, training=network.training)
        anchor_vec = node_vec[sampled_node_idx]

        first_node_anchor_adj, first_anchor_adj = cur_node_anchor_adj, cur_anchor_adj
        first_init_agg_vec = network.encoder.graph_encoders[0](init_node_vec, first_node_anchor_adj, anchor_mp=True, batch_norm=False)

        # Add mid GNN layers
        for encoder in network.encoder.graph_encoders[1:-1]:
            node_vec = (1 - network.graph_skip_conn) * encoder(node_vec, cur_node_anchor_adj, anchor_mp=True, batch_norm=False) + \
                        network.graph_skip_conn * encoder(node_vec, init_adj, anchor_mp=False, batch_norm=False)

            if encoder.bn is not None:
                node_vec = encoder.compute_bn(node_vec)

            node_vec = torch.relu(node_vec)
            node_vec = F.dropout(node_vec, network.dropout, training=network.training)
            anchor_vec = node_vec[sampled_node_idx]

        # Compute output via node-anchor-node message passing
        output = (1 - network.graph_skip_conn) * network.encoder.graph_encoders[-1](node_vec, cur_node_anchor_adj, anchor_mp=True, batch_norm=False) + \
                    network.graph_skip_conn * network.encoder.graph_encoders[-1](node_vec, init_adj, anchor_mp=False, batch_norm=False)
        output = F.log_softmax(output, dim=-1)
        score = self.model.score_func(labels[idx], output[idx])
        loss1 = self.model.criterion(output[idx], labels[idx])

        if self.config['graph_learn'] and self.config['graph_learn_regularization']:
            loss1 += self.add_graph_loss(cur_anchor_adj, init_anchor_vec)

        if not mode == 'test':
            if self._epoch > self.config.get('pretrain_epoch', 0):
                max_iter_ = self.config.get('max_iter', 10) # Fine-tuning
                if self._epoch == self.config.get('pretrain_epoch', 0) + 1:
                    for k in self._dev_metrics:
                        self._best_metrics[k] = -float('inf')

            else:
                max_iter_ = 0 # Pretraining
        else:
            max_iter_ = self.config.get('max_iter', 10)

        if training:
            eps_adj = float(self.config.get('eps_adj', 0)) # cora: 5.5e-8, cora w/o input graph: 1e-8, citeseer w/o input graph: 1e-8, wine: 2e-5, cancer: 2e-5, digtis: 2e-5
        else:
            eps_adj = float(self.config.get('test_eps_adj', self.config.get('eps_adj', 0)))

        pre_node_anchor_adj = cur_node_anchor_adj

        loss = 0
        iter_ = 0
        while self.config['graph_learn'] and (iter_ == 0 or diff(cur_node_anchor_adj, pre_node_anchor_adj, cur_node_anchor_adj).item() > eps_adj) and iter_ < max_iter_:
            iter_ += 1
            pre_node_anchor_adj = cur_node_anchor_adj

            # Compute n x s node-anchor relationship matrix
            cur_node_anchor_adj = network.learn_graph(network.graph_learner2, node_vec, anchor_features=anchor_vec)

            # Compute s x s anchor graph
            cur_anchor_adj = compute_anchor_adj(cur_node_anchor_adj)

            cur_agg_vec = network.encoder.graph_encoders[0](init_node_vec, cur_node_anchor_adj, anchor_mp=True, batch_norm=False)

            update_adj_ratio = self.config.get('update_adj_ratio', None)
            if update_adj_ratio is not None:
                cur_agg_vec = update_adj_ratio * cur_agg_vec + (1 - update_adj_ratio) * first_init_agg_vec

            node_vec = (1 - network.graph_skip_conn) * cur_agg_vec + \
                    network.graph_skip_conn * init_agg_vec

            if network.encoder.graph_encoders[0].bn is not None:
                node_vec = network.encoder.graph_encoders[0].compute_bn(node_vec)

            node_vec = torch.relu(node_vec)
            node_vec = F.dropout(node_vec, self.config.get('gl_dropout', 0), training=network.training)
            anchor_vec = node_vec[sampled_node_idx]

            # Add mid GNN layers
            for encoder in network.encoder.graph_encoders[1:-1]:
                mid_cur_agg_vec = encoder(node_vec, cur_node_anchor_adj, anchor_mp=True, batch_norm=False)
                if update_adj_ratio is not None:
                    mid_first_agg_vecc = encoder(node_vec, first_node_anchor_adj, anchor_mp=True, batch_norm=False)
                    mid_cur_agg_vec = update_adj_ratio * mid_cur_agg_vec + (1 - update_adj_ratio) * mid_first_agg_vecc

                node_vec = (1 - network.graph_skip_conn) * mid_cur_agg_vec + \
                        network.graph_skip_conn * encoder(node_vec, init_adj, anchor_mp=False, batch_norm=False)

                if encoder.bn is not None:
                    node_vec = encoder.compute_bn(node_vec)

                node_vec = torch.relu(node_vec)
                node_vec = F.dropout(node_vec, self.config.get('gl_dropout', 0), training=network.training)
                anchor_vec = node_vec[sampled_node_idx]

            cur_agg_vec = network.encoder.graph_encoders[-1](node_vec, cur_node_anchor_adj, anchor_mp=True, batch_norm=False)
            if update_adj_ratio is not None:
                first_agg_vec = network.encoder.graph_encoders[-1](node_vec, first_node_anchor_adj, anchor_mp=True, batch_norm=False)
                cur_agg_vec = update_adj_ratio * cur_agg_vec + (1 - update_adj_ratio) * first_agg_vec

            output = (1 - network.graph_skip_conn) * cur_agg_vec + \
                    network.graph_skip_conn * network.encoder.graph_encoders[-1](node_vec, init_adj, anchor_mp=False, batch_norm=False)

            output = F.log_softmax(output, dim=-1)
            score = self.model.score_func(labels[idx], output[idx])
            loss += self.model.criterion(output[idx], labels[idx])

            if self.config['graph_learn'] and self.config['graph_learn_regularization']:
                loss += self.add_graph_loss(cur_anchor_adj, init_anchor_vec)

            if self.config['graph_learn'] and not self.config.get('graph_learn_ratio', None) in (None, 0):
                loss += SquaredFrobeniusNorm(cur_node_anchor_adj - pre_node_anchor_adj) * self.config.get('graph_learn_ratio')

        if mode == 'test' and self.config.get('out_raw_learned_adj_path', None):
            out_raw_learned_adj_path = os.path.join(self.dirname, self.config['out_raw_learned_adj_path'])
            np.save(out_raw_learned_adj_path, cur_node_anchor_adj.cpu())
            print('Saved raw_learned_adj to {}'.format(out_raw_learned_adj_path))

        if iter_ > 0:
            loss = loss / iter_ + loss1
        else:
            loss = loss1
        if training:
            self.model.optimizer.zero_grad()
            loss.backward()
            self.model.clip_grad()
            self.model.optimizer.step()

        self._update_metrics(loss.item(), {'nloss': -loss.item(), self.model.metric_name: score}, 1, training=training)
        return output[idx], labels[idx]

    # 跑一轮的图分类----------------------
    def _run_batch_epoch(self, data_loader, training=True, test=False, rl_ratio=0, verbose=10, out_predictions=False, out_adj=False):
        start_time = time.time()
        mode = "train" if training else ("test" if self.is_test else "dev")

        if training:
            self.model.optimizer.zero_grad()

        # 输出以及得分
        output = []
        gold = []
        idx = []
        init_adj = []
        opt_adj = []
        init_adj_all = []
        opt_adj_all = []
        y_prod_all = []
        y_pred_all = []
        y_true_all = []
        vectorize_input = vectorize_input_eeg if self.data_type == 'eeg' else vectorize_input_text
        with torch.enable_grad(),tqdm(total=data_loader.get_num_instance()) as progress_bar:
            # data_loader：传进来的是train_loader或者其他的，在init中定义，再深一点就是在DataStream
            for step in range(data_loader.get_num_batch()):
                # 获取此batch的输入数据
                input_batch = data_loader.nextBatch()
                x_batch = vectorize_input(input_batch, self.config, training=training, device=self.device)
                if not x_batch:
                    continue  # When there are no examples in the batch

                # 不用gnn
                if self.config.get('no_gnn', False):
                    res = self.batch_no_gnn(x_batch, step, training=training, out_predictions=out_predictions)
                else:
                    batch_IGL_stop = self.eeg_batch_IGL_stop if self.data_type == 'eeg' else self.batch_IGL_stop
                    # 使用锚点集
                    if self.config.get('scalable_run', False):
                        res = self.scalable_batch_IGL_stop(x_batch, step, training=training, out_predictions=out_predictions)
                    # 重点在这里batch_IGL_stop！！！
                    else:
                        res, logits = batch_IGL_stop(x_batch, step, training=training, out_predictions=out_predictions, out_adj=out_adj)

                    # (batch_size, num_classes)
                    y_prod = F.softmax(logits, dim=1).detach().cpu().numpy()
                    y_pred = np.argmax(y_prod, axis=1).reshape(-1)  # (batch_size,)
                    y_true = x_batch['targets'].cpu().numpy().astype(int)
                    if self.config['graph_module'] == 'ggnn' and self.config['graph_learn']:
                        res_init_adj = res['init_adj'].numpy()
                        res_opt_adj = res['opt_adj'].numpy()
                y_prod_all.append(y_prod)
                y_pred_all.append(y_pred)
                y_true_all.append(y_true)
                if self.config['graph_module'] == 'ggnn' and self.config['graph_learn']:
                    init_adj_all.append(res_init_adj)
                    opt_adj_all.append(res_opt_adj)

                loss = res['loss']
                if training:
                    self._train_loss.update(loss)
                    self._train_metrics['nloss'].update(-loss, x_batch['batch_size'])
                else:
                    self._dev_loss.update(loss)
                    self._dev_metrics['nloss'].update(-loss, x_batch['batch_size'])

                if training:
                    self._n_train_examples += x_batch['batch_size']

                if (verbose > 0) and (step > 0) and (step % verbose == 0):
                    summary_str = self.self_report(step, mode)
                    self.logger.write_to_file(summary_str)
                    print(summary_str)
                    print('used_time: {:0.2f}s'.format(time.time() - start_time))

                if not training and out_predictions:
                    output.extend(res['predictions'])
                    gold.extend(x_batch['targets'])
                    idx.extend(x_batch['idx'])

                if not training and out_adj:
                    init_adj.extend(res['init_adj'])
                    opt_adj.extend(res['opt_adj'])

                progress_bar.update(x_batch['batch_size'])

        y_prod_all = np.concatenate(y_prod_all, axis=0)
        y_pred_all = np.concatenate(y_pred_all, axis=0)
        y_true_all = np.concatenate(y_true_all, axis=0)
        if self.config['graph_module'] == 'ggnn' and self.config['graph_learn']:
            init_adj_all = np.concatenate(init_adj_all, axis=0)
            opt_adj_all = np.concatenate(opt_adj_all, axis=0)

        score_acc = self.model.score_func(y_true_all, y_pred_all)
        scores_dict = self.model.score_func2(y_true_all, y_pred_all)
        output_auc = score_acc

        metrics = {self.model.metric_name: score_acc,
                    self.model.metric_name_precision: scores_dict['precision'],
                    self.model.metric_name_recall: scores_dict['recall'],
                    self.model.metric_name_f1: scores_dict['f1'],
                    self.model.metric_name_Weight_F1: scores_dict['Weighted_F1']
                    }

        # 计算每个类别的混淆矩阵
        conf_matrices = []
        for class_label in range(4):  # 请替换为实际的类别数量
            y_true_class = (y_true_all == class_label).astype(int)
            y_pred_class = (y_pred_all == class_label).astype(int)
            conf_matrix_class = confusion_matrix(y_true_class, y_pred_class)
            conf_matrices.append(conf_matrix_class)

        # 构建格式化字符串并打印混淆矩阵
        format_str_conf_matrix = "\nConfusion Matrix for each class:"
        for class_label, conf_matrix_class in enumerate(conf_matrices):
            format_str_conf_matrix += f"\n\nClass {class_label} Confusion Matrix:\n{conf_matrix_class}"
        print(format_str_conf_matrix)
        self.logger.write_to_file(format_str_conf_matrix)

        # 计算每个类别的 AUROC
        auroc_dict = {}
        for class_label in range(4):  # 请替换为实际的类别数量
            y_true_class = (y_true_all == class_label).astype(int)
            y_prob_class = y_prod_all[:, class_label]
            auroc = roc_auc_score(y_true_class, y_prob_class)
            auroc_dict[f'Class_{class_label}_AUROC'] = auroc

        # 构建格式化字符串并打印
        format_str_auroc = "\nAUROC for each class:"
        for class_label, auroc in auroc_dict.items():
            format_str_auroc += f"\nClass {class_label}: AUROC = {auroc}"
        print(format_str_auroc)
        self.logger.write_to_file(format_str_auroc)

        if training:
            for k in self._train_metrics:
                if not k in metrics:
                    continue
                self._train_metrics[k] = metrics[k]

        else:
            for k in self._dev_metrics:
                if not k in metrics:
                    continue
                self._dev_metrics[k] = metrics[k]

        if not training and out_adj:
            return output, gold, idx, init_adj, opt_adj, score_acc

        if test:
            # 保存 y_prod_all、y_pred_all 和 y_true_all
            save_path = r'F:\U盘\IDGL-master\out\tusz_eeg\4_class\GGNN\new'
            np.save(os.path.join(save_path, 'y_prod_all.npy'), y_prod_all)
            np.save(os.path.join(save_path, 'y_pred_all.npy'), y_pred_all)
            np.save(os.path.join(save_path, 'y_true_all.npy'), y_true_all)
            if self.config['graph_module'] == 'ggnn' and self.config['graph_learn']:
                np.save(os.path.join(save_path, 'init_adj_all.npy'), init_adj_all)
                np.save(os.path.join(save_path, 'opt_adj_all.npy'), opt_adj_all)

        return output, gold, idx, output_auc

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def self_report(self, step, mode='train'):
        if mode == "train":
            format_str = "[train-{}] step: [{} / {}] | loss = {:0.5f}".format(
                self._epoch, step, self._n_train_batches, self._train_loss.mean())
            format_str += self.metric_to_str(self._train_metrics)
        elif mode == "dev":
            format_str = "[predict-{}] step: [{} / {}] | loss = {:0.5f}".format(
                    self._epoch, step, self._n_dev_batches, self._dev_loss.mean())
            format_str += self.metric_to_str(self._dev_metrics)
        elif mode == "test":
            format_str = "[test] | test_exs = {} | step: [{} / {}]".format(
                    self._n_test_examples, step, self._n_test_batches)
            format_str += self.metric_to_str(self._dev_metrics)
        else:
            raise ValueError('mode = {} not supported.' % mode)
        return format_str

    def plain_metric_to_str(self, metrics):
        format_str = ''
        for k in metrics:
            format_str += ' | {} = {:0.5f}'.format(k.upper(), metrics[k])
        return format_str

    # 将指标转化成字符串的格式化函数
    def metric_to_str(self, metrics):
        format_str = ''
        for k in metrics:
            if k == 'nloss':
                format_str += ' | {} = {:0.5f}'.format(k.upper(), metrics[k].mean())
            else:
                format_str += ' | {} = {:0.5f}'.format(k.upper(), metrics[k])
        return format_str

    # 将最好的指标转化成字符串的格式化函数
    def best_metric_to_str(self, metrics):
        format_str = '\n'
        for k in metrics:
            if k == 'nloss':
                format_str += ' | {} = {:0.5f}'.format(k.upper(), metrics[k].mean())
            else:
                format_str += ' | {} = {:0.5f}'.format(k.upper(), metrics[k])
        return format_str

    def summary(self):
        start = "\n<<<<<<<<<<<<<<<< MODEL SUMMARY >>>>>>>>>>>>>>>> "
        info = "Best epoch = {}; ".format(self._best_epoch) + self.best_metric_to_str(self._best_metrics)
        end = " <<<<<<<<<<<<<<<< MODEL SUMMARY >>>>>>>>>>>>>>>> "
        return "\n".join([start, info, end])

    # 更新指标
    def _update_metrics(self, loss, metrics, batch_size, training=True):
        if training:
            if loss:
                self._train_loss.update(loss)
            for k in self._train_metrics:
                if not k in metrics:
                    continue
                self._train_metrics[k].update(metrics[k], batch_size)
        else:
            if loss:
                self._dev_loss.update(loss)
            for k in self._dev_metrics:
                if not k in metrics:
                    continue
                self._dev_metrics[k].update(metrics[k], batch_size)

    # 重置指标（训练集、验证集）
    def _reset_metrics(self):
        self._train_loss.reset()
        self._dev_loss.reset()

        for k in self._train_metrics:
            if k == 'nloss':
                self._train_metrics[k].reset()
            else:
                self._train_metrics[k] = -float('inf')
        for k in self._dev_metrics:
            if k == 'nloss':
                self._dev_metrics[k].reset()
            else:
                self._dev_metrics[k] = -float('inf')

    # 检查没有超过最大训练epoch，也没有达到没有改进的耐心周期。
    def _stop_condition(self, epoch, patience=10):
        """
        Checks have not exceeded max epochs and has not gone patience epochs without improvement.
        检查没有超过最大周期，也没有达到没有改进的耐心周期。
        """
        no_improvement = epoch >= self._best_epoch + patience
        exceeded_max_epochs = epoch >= self.config['max_epochs']
        return False if exceeded_max_epochs or no_improvement else True

    def add_graph_loss(self, out_adj, features):
        # Graph regularization
        graph_loss = 0
        L = torch.diagflat(torch.sum(out_adj, -1)) - out_adj
        graph_loss += self.config['smoothness_ratio'] * torch.trace(torch.mm(features.transpose(-1, -2), torch.mm(L, features))) / int(np.prod(out_adj.shape))
        ones_vec = to_cuda(torch.ones(out_adj.size(-1)), self.device)
        graph_loss += -self.config['degree_ratio'] * torch.mm(ones_vec.unsqueeze(0), torch.log(torch.mm(out_adj, ones_vec.unsqueeze(-1)) + Constants.VERY_SMALL_NUMBER)).squeeze() / out_adj.shape[-1]
        graph_loss += self.config['sparsity_ratio'] * torch.sum(torch.pow(out_adj, 2)) / int(np.prod(out_adj.shape))
        return graph_loss

    # 图正则化
    def add_batch_graph_loss(self, out_adj, features, keep_batch_dim=False):
        # Graph regularization
        if keep_batch_dim:
            graph_loss = []
            for i in range(out_adj.shape[0]):
                L = torch.diagflat(torch.sum(out_adj[i], -1)) - out_adj[i]
                graph_loss.append(self.config['smoothness_ratio'] * torch.trace(torch.mm(features[i].transpose(-1, -2), torch.mm(L, features[i]))) / int(np.prod(out_adj.shape[1:])))

            graph_loss = to_cuda(torch.Tensor(graph_loss), self.device)

            ones_vec = to_cuda(torch.ones(out_adj.shape[:-1]), self.device)
            # 计算每个节点的度的对数
            log_degrees = torch.log(torch.matmul(out_adj, ones_vec.unsqueeze(-1)) + Constants.VERY_SMALL_NUMBER)

            # 计算总图度
            total_degrees = torch.matmul(ones_vec.unsqueeze(1), log_degrees).squeeze(-1).squeeze(-1)

            # 计算度正则化项
            degree_regularization = -self.config['degree_ratio'] * total_degrees / out_adj.shape[-1]

            # 将度正则化项添加到图损失中
            graph_loss += degree_regularization
            graph_loss += self.config['sparsity_ratio'] * torch.sum(torch.pow(out_adj, 2), (1, 2)) / int(np.prod(out_adj.shape[1:]))

        else:
            graph_loss = 0
            for i in range(out_adj.shape[0]):
                L = torch.diagflat(torch.sum(out_adj[i], -1)) - out_adj[i]
                graph_loss += self.config['smoothness_ratio'] * torch.trace(torch.mm(features[i].transpose(-1, -2), torch.mm(L, features[i]))) / int(np.prod(out_adj.shape))

            ones_vec = to_cuda(torch.ones(out_adj.shape[:-1]), self.device)
            graph_loss += -self.config['degree_ratio'] * torch.matmul(ones_vec.unsqueeze(1), torch.log(torch.matmul(out_adj, ones_vec.unsqueeze(-1)) + Constants.VERY_SMALL_NUMBER)).sum() / out_adj.shape[0] / out_adj.shape[-1]
            graph_loss += self.config['sparsity_ratio'] * torch.sum(torch.pow(out_adj, 2)) / int(np.prod(out_adj.shape))

        return graph_loss


# 定义评价指标
# 计算准确率的数据结构
stats_train_data = {}
stats_val_data = {}
stats_test_data = {}
# --- 1_Train ---
auroc_patient_train_folds = []  # ROC曲线下的面积(患者级别)
auroc_train_folds = []  # ROC曲线下的面积(窗口级别)
precision_patient_train_folds = []  # 查准率
recall_patient_train_folds = []  # 查全率
f1_patient_train_folds = []  # F1 score
bal_acc_patient_train_folds = []  # 正确率
# --- val ---
auroc_patient_val_folds = []  # ROC曲线下的面积(患者级别)
auroc_val_folds = []  # ROC曲线下的面积(窗口级别)
precision_patient_val_folds = []  # 查准率
recall_patient_val_folds = []  # 查全率
f1_patient_val_folds = []  # F1 score
bal_acc_patient_val_folds = []  # 正确率
# --- test ---
auroc_patient_test_folds = []  # ROC曲线下的面积(患者级别)
auroc_test_folds = []  # ROC曲线下的面积(窗口级别)
precision_patient_test_folds = []  # 查准率
recall_patient_test_folds = []  # 查全率
f1_patient_test_folds = []  # F1 score
bal_acc_patient_test_folds = []  # 正确率

y_probs_ = []
y_true_ = []


# 新的计算准确率的函数
def collect_metrics_new(y_probs, y_true, sample_indices, fold_idx, stats_index_data,
                        experiment_type, dataset_index, y_path):
    # y_probs：模型的输出
    # y_true：真实值
    # sample_indices：样本的索引
    # fold_idx：训练的轮数
    # stats_index_data：计算准确率的数据结构
    # 获取每个窗口的在源数据的相关信息

    # 创建训练和测试表
    rows = []
    y_true = y_true.tolist()
    y_probs = F.softmax(y_probs, dim=1)
    y_preds = torch.argmax(y_probs, dim=1)
    y_preds = y_preds.tolist()
    sample_indices = sample_indices.tolist()
    for i in range(len(sample_indices)):
        idx = sample_indices[i]
        temp = {}
        temp["patient_ID"] = str(dataset_index.loc[idx, "patient_ID"])
        temp["sample_idx"] = idx
        temp["y_true"] = y_true[i]
        temp["y_probs"] = y_preds[i]
        rows.append(temp)
    patient_df = pd.DataFrame(rows)

    # 检查patient的所有标签
    # patient_id_yu = set(list(patient_df["patient_ID"]))
    # group_yu = patient_df.groupby("patient_ID")
    # for i in patient_id_yu:
    #     print("patient_ID为", i, "的所有标签为", set(group_yu.get_group(i)["y_true"]))

    # get patient-level metrics from window-level dataframes
    # 统计患者级的相关矩阵
    y_probs_patient, y_true_patient = get_patient_prediction_new(patient_df, fold_idx, stats_index_data, experiment_type)
    # 保存 y_true & y_probs
    # save_y_probs(y_true_patient, y_probs_patient)

    stats_index_data[f"probs_fold_{fold_idx}"] = y_probs_patient
    # 保存结果用于画图
    if experiment_type == "test":
        save_y_probs(y_true, y_preds, y_path)
        # 保存最后一次的y_true & y_probs
        y_true_.extend(y_true_patient)
        y_probs_.extend(y_probs_patient)
        save_y_probs_patient(y_true_, y_probs_, y_path)

    # 定义两个字典保存窗口以及患者级别的参数
    window_csv_dict = {}
    patient_csv_dict = {}

    # WINDOW-LEVEL ROC PLOT
    # pos_label="healthy" & 1
    # fpr：数组，随阈值上涨的假阳性率
    # tpr：数组，随阈值上涨的真正例率
    # thresholds：数组，对预测值排序后的score列表，作为阈值，排序从大到小
    fpr, tpr, thresholds = roc_curve(y_true_patient, y_probs_patient, pos_label=1)
    # fpr, tpr, thresholds = roc_curve(y_true_patient, y_probs_patient[:, 1], pos_label=1)
    # fpr, tpr, thresholds = roc_curve(y_true_test_patient, y_probs_test_patient[:,0], pos_label=0)
    # patient_csv_dict[f"fpr_fold_{fold_idx}"] = fpr
    # patient_csv_dict[f"tpr_fold_{fold_idx}"] = tpr
    # patient_csv_dict[f"thres_fold_{fold_idx}"] = thresholds

    # select an optimal threshold using the ROC curve Youden's J statistic to obtain the optimal probability
    # threshold and this method gives equal weights to both false positives and false negatives
    # 使用ROC曲线Youden's J统计量选择一个最佳阈值，以获得最佳概率
    # threshold，这个方法对误报和误报都给予相等的权重
    optimal_proba_cutoff = sorted(list(zip(np.abs(tpr - fpr), thresholds)), key=lambda i: i[0], reverse=True)[0][1]
    # print("概率分类最佳阈值：", optimal_proba_cutoff)

    # calculate class predictions and confusion-based metrics using the optimal threshold
    # 使用最佳阈值计算类预测和基于混淆的度量
    roc_predictions = [1 if i >= optimal_proba_cutoff else 0 for i in y_probs_patient]
    # roc_predictions = [1 if i >= optimal_proba_cutoff else 0 for i in y_probs_patient[:, 1]]
    # roc_predictions = [0 if i >= optimal_proba_cutoff else 1 for i in y_probs_test_patient[:,0]]

    precision_patient = precision_score(y_true_patient, roc_predictions)
    recall_patient = recall_score(y_true_patient, roc_predictions)
    f1_patient = f1_score(y_true_patient, roc_predictions)
    bal_acc_patient = balanced_accuracy_score(y_true_patient, roc_predictions)
    from sklearn.metrics import roc_auc_score
    auroc_patient = roc_auc_score(y_true_patient, y_probs_patient)

    return auroc_patient, precision_patient, recall_patient, f1_patient, bal_acc_patient


# 创建患者级矩阵
def get_patient_prediction_new(df, fold_idx, stats_temp_data, experiment_type):
    # 患者的个数
    unique_patients = list(df["patient_ID"].unique())
    # 将不同的患者分组，（因为一个患者的脑电波会分成多个窗口）
    grouped_df = df.groupby("patient_ID")
    rows = []
    # 逐个患者循环
    for patient in unique_patients:
        patient_df = grouped_df.get_group(patient)
        temp = {}
        temp["patient_ID"] = patient
        temp["y_true"] = list(patient_df["y_true"].unique())[0]  # 真实标签
        assert len(list(patient_df["y_true"].unique())) == 1
        temp["y_probs"] = patient_df["y_probs"].mean()
        rows.append(temp)
    return_df = pd.DataFrame(rows)

    # need subject names and labels for comparisons testing
    if experiment_type == "test" and fold_idx == 0:
        stats_temp_data["subject_id"] = list(return_df["patient_ID"][:])
        stats_temp_data["label"] = return_df["y_true"][:]

    # 返回值：
    return list(return_df["y_probs"]), list(return_df["y_true"])


# 保存 y_true & y_probs 用于画ROC曲线图
def save_y_probs(a, b, path):
    y_true_path = path + '/y_true.npy'
    y_probs_path = path + '/y_probs.npy'
    np.save(y_true_path, a)
    np.save(y_probs_path, b)


# 保存 y_true_patient & y_probs_patient 用于画patient级别ROC曲线图
def save_y_probs_patient(a, b, path):
    y_true_patient_path = path + '/y_true_patient.npy'
    y_probs_patient_path = path + '/y_probs_patient.npy'
    np.save(y_true_patient_path, a)
    np.save(y_probs_patient_path, b)


# 保存训练前后的adj
def save_adj(a, b, path):
    init_adj_path = path + '/init_adj.npy'
    opt_adj_path = path + '/opt_adj.npy'
    np.save(init_adj_path, a)
    np.save(opt_adj_path, b)


def diff(X, Y, Z):
    assert X.shape == Y.shape
    # pow(X - Y, 2)   ---   平方
    diff_ = torch.sum(torch.pow(X - Y, 2))
    norm_ = torch.sum(torch.pow(Z, 2))
    # 将输入input张量每个元素的范围限制到区间[min, norm_]
    diff_ = diff_ / torch.clamp(norm_, min=Constants.VERY_SMALL_NUMBER)
    return diff_


def batch_diff(X, Y, Z):
    assert X.shape == Y.shape
    diff_ = torch.sum(torch.pow(X - Y, 2), (1, 2)) # Shape: [batch_size]
    norm_ = torch.sum(torch.pow(Z, 2), (1, 2))
    diff_ = diff_ / torch.clamp(norm_, min=Constants.VERY_SMALL_NUMBER)
    return diff_


def SquaredFrobeniusNorm(X):
    return torch.sum(torch.pow(X, 2)) / int(np.prod(X.shape))


def batch_SquaredFrobeniusNorm(X):
    return torch.sum(torch.pow(X, 2), (1, 2)) / int(np.prod(X.shape[1:]))

