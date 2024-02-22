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
        self._train_loss = AverageMeter()
        self._dev_loss = AverageMeter()
        self.data_type = config['data_type']
        if config['dataset_name'] == 'TUEP_EEG':
            self.dataset_index = pd.read_csv(config['index_file_path'], dtype={"patient_ID": str})
        if config['dataset_name'] == 'TUAB_EEG':
            self.dataset_index = pd.read_csv(config['index_file_path'], dtype={"patient_ID": str}, index_col=0)
        if config['dataset_name'] == 'TUSZ_EEG':
            self.dataset_index = pd.read_csv(config['index_file_path'], dtype={"patient_ID": str}, index_col=0)

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

        elif config['task_type'] == 'regression':
            self._train_metrics = {'nloss': AverageMeter(),
                                   'r2': AverageMeter()}
            self._dev_metrics = {'nloss': AverageMeter(),
                                 'r2': AverageMeter()}

        else:
            raise ValueError('Unknown task_type: {}'.format(config['task_type']))
        self.logger = DummyLogger(config, dirname=config['out_dir'], pretrained=config['pretrained'])
        self.dirname = self.logger.dirname
        if not config['no_cuda'] and torch.cuda.is_available():
            print('[ Using CUDA ]')
            self.device = torch.device('cuda' if config['cuda_id'] < 0 else 'cuda:%d' % config['cuda_id'])
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')
        config['device'] = self.device

        seed = config.get('seed', 42)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.device:
            torch.cuda.manual_seed(seed)

        datasets = prepare_datasets(config)

        # Prepare datasets
        # network:Cora、Citeseer、Pubmed、ogbn-arxiv
        # uci: Wine、Cancer、Digits
        if config['data_type'] == "eeg":
            train_set = datasets['train']
            dev_set = datasets['dev']
            test_set = datasets['test']

            self.run_epoch = self._run_batch_epoch

            # Initialize the model
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
        timer = Timer("Train")
        self._epoch = self._best_epoch = 0
        self._best_metrics = {}
        for k in self._dev_metrics:
            self._best_metrics[k] = -float('inf')
        self._reset_metrics()

        while self._stop_condition(self._epoch, self.config['patience']):
            self._epoch += 1

            # Train phase
            if self._epoch % self.config['print_every_epochs'] == 0:
                format_str = "\n>>> Train Epoch: [{} / {}]".format(self._epoch, self.config['max_epochs'])
                print(format_str)
                self.logger.write_to_file(format_str)

            self.run_epoch(self.train_loader, training=True, test=False, verbose=self.config['verbose'])

            # When print_every_epochs=1 print
            if self._epoch % self.config['print_every_epochs'] == 0:
                format_str = "Training Epoch {} -- Loss: {:0.5f}".format(self._epoch, self._train_loss.mean())
                format_str += self.metric_to_str(self._train_metrics)
                train_epoch_time_msg = timer.interval("Training Epoch {}".format(self._epoch))
                self.logger.write_to_file(train_epoch_time_msg + '\n' + format_str)
                print(format_str)
                format_str = "\n>>> Validation Epoch: [{} / {}]".format(self._epoch, self.config['max_epochs'])
                print(format_str)
                self.logger.write_to_file(format_str)

            # Validation phase
            dev_output, dev_gold, dev_idx, score_auc = self.run_epoch(self.dev_loader, training=False, test=False, verbose=self.config['verbose']
                                                           , out_predictions=self.config['out_predictions'])
            if self.config['out_predictions']:
                dev_output = torch.stack(dev_output, 0)
                dev_gold = torch.stack(dev_gold, 0)
                # 计算acc
                dev_metric_score = score_auc
            else:
                dev_metric_score = None

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

            if not self.config['data_type'] in ('network', 'uci', 'text', 'eeg'):
                self.model.scheduler.step(self._dev_metrics[self.config['eary_stop_metric']].mean())

            cur_dev_score = self._dev_metrics[self.config['eary_stop_metric']]


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

            # reset metrics
            self._reset_metrics()

        timer.finish()

        format_str = "Finished Training: {}\nTraining time: {}".format(self.dirname, timer.total) + '\n' + self.summary()
        print(format_str)
        self.logger.write_to_file(format_str)
        return self._best_metrics

    def test(self):
        if self.test_loader is None:
            print("No testing set specified -- skipped testing.")
            return

        # Restore best model
        print('Restoring best model')
        # 加载已经保存的模型
        self.model.init_saved_network(self.dirname)
        self.model.network = self.model.network.to(self.device)

        self.is_test = True
        self._reset_metrics()
        timer = Timer("Test")

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


    # Iterative of eeg
    def eeg_batch_IGL_stop(self, x_batch, step, training, out_predictions=False, out_adj=False):
        '''Iterative graph learning: batch training, batch stopping'''
        mode = "train" if training else ("test" if self.is_test else "dev")
        network = self.model.network
        network.train(training)

        context, targets, adj = x_batch['context'], x_batch['targets'], x_batch['adj']
        context_lens = x_batch['context_lens']
        seq_lengths = self.config['min_word_freq']
        seq_lengths = np.full((x_batch['batch_size'],), fill_value=seq_lengths)



        # Prepare init node embedding, init adj
        # raw_context_vec, context_vec, context_mask, init_adj
        raw_context_vec, context_vec, init_adj = network.prepare_init_graph(context, context_lens, adj)

        # Init
        raw_node_vec = raw_context_vec # word embedding
        init_node_vec = context_vec # hidden embedding
        node_mask = None

        if self.config['graph_learn']:
            cur_raw_adj, cur_adj = network.learn_graph(network.graph_learner, raw_node_vec, network.graph_skip_conn, node_mask=node_mask, graph_include_self=network.graph_include_self, init_adj=init_adj)

        if self.config['graph_module'] == 'gcn':
            # GCN
            node_vec = torch.relu(network.encoder.graph_encoders[0](init_node_vec, cur_adj))
            node_vec = F.dropout(node_vec, network.dropout, training=network.training)

            # Add mid GNN layers
            for encoder in network.encoder.graph_encoders[1:-1]:
                node_vec = torch.relu(encoder(node_vec, cur_adj))
                node_vec = F.dropout(node_vec, network.dropout, training=network.training)

            # BP to update weights
            output = network.encoder.graph_encoders[-1](node_vec, cur_adj)
            output = network.compute_output(output, node_mask=node_mask)
        elif self.config['graph_module'] == 'ggnn': # Gated Graph Neural Networks
            node_vec = torch.relu(network.encoder(init_node_vec, cur_adj))
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

            # GGNN
            node_vec = torch.relu(network.encoder(init_node_vec, cur_adj))
            tmp_output = network.compute_output(node_vec, node_mask=node_mask)
            batch_all_outputs.append(tmp_output.unsqueeze(1))

            tmp_loss = self.model.criterion(tmp_output, targets)
            if len(tmp_loss.shape) == 2:
                tmp_loss = torch.mean(tmp_loss, 1)

            loss += batch_stop_indicators.float() * tmp_loss

            # Add Graph Loss
            if self.config['graph_learn'] and self.config['graph_learn_regularization']:
                graph_loss = batch_stop_indicators.float() * self.add_batch_graph_loss(cur_raw_adj, raw_node_vec, keep_batch_dim=True)
                loss += graph_loss

            if self.config['graph_learn'] and not self.config.get('graph_learn_ratio', None) in (None, 0):
                loss += batch_stop_indicators.float() * batch_SquaredFrobeniusNorm(cur_raw_adj - pre_raw_adj) * self.config.get('graph_learn_ratio')

            tmp_stop_criteria = batch_diff(cur_raw_adj, pre_raw_adj, first_raw_adj) > eps_adj
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


        else:
            loss = loss1

        if training:
            res = {'loss': loss.item(),
                   }
        else:
            res = {'loss': loss.item(),
                   }
        if out_predictions:
            res['predictions'] = output.detach().cpu()
            res['idx'] = x_batch['idx'].cpu()
        # save new_adj, init_adj
        if out_adj:
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


    # Run round of graph classification
    def _run_batch_epoch(self, data_loader, training=True, test=False, rl_ratio=0, verbose=10, out_predictions=False, out_adj=False):
        start_time = time.time()
        mode = "train" if training else ("test" if self.is_test else "dev")

        if training:
            self.model.optimizer.zero_grad()

        output = []
        gold = []
        idx = []
        init_adj = []
        opt_adj = []
        y_prod_all = []
        y_pred_all = []
        y_true_all = []
        vectorize_input = vectorize_input_eeg if self.data_type == 'eeg' else vectorize_input_text
        with torch.enable_grad(),tqdm(total=data_loader.get_num_instance()) as progress_bar:
            for step in range(data_loader.get_num_batch()):
                input_batch = data_loader.nextBatch()
                x_batch = vectorize_input(input_batch, self.config, training=training, device=self.device)
                if not x_batch:
                    continue  # When there are no examples in the batch

                    batch_IGL_stop = self.eeg_batch_IGL_stop if self.data_type == 'eeg' else self.batch_IGL_stop
                    res, logits = batch_IGL_stop(x_batch, step, training=training, out_predictions=out_predictions, out_adj=out_adj)

                    # (batch_size, num_classes)
                    y_prod = F.softmax(logits, dim=1).detach().cpu().numpy()
                    y_pred = np.argmax(y_prod, axis=1).reshape(-1)  # (batch_size,)
                    y_true = x_batch['targets'].cpu().numpy().astype(int)

                y_prod_all.append(y_prod)
                y_pred_all.append(y_pred)
                y_true_all.append(y_true)

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

        score_acc = self.model.score_func(y_true_all, y_pred_all)
        scores_dict = self.model.score_func2(y_true_all, y_pred_all)
        output_auc = score_acc

        metrics = {self.model.metric_name: score_acc,
                    self.model.metric_name_precision: scores_dict['precision'],
                    self.model.metric_name_recall: scores_dict['recall'],
                    self.model.metric_name_f1: scores_dict['f1'],
                    self.model.metric_name_Weight_F1: scores_dict['Weighted_F1']
                    }

        # Calculate the AUROC for each category
        auroc_dict = {}
        for class_label in range(4):
            y_true_class = (y_true_all == class_label).astype(int)
            y_prob_class = y_prod_all[:, class_label]
            auroc = roc_auc_score(y_true_class, y_prob_class)
            auroc_dict[f'Class_{class_label}_AUROC'] = auroc

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
            # save y_prod_all\y_pred_all\y_true_all
            save_path = r'E:\IDGL-master\out\tusz_eeg\4_class\LSTM'
            np.save(os.path.join(save_path, 'y_prod_all.npy'), y_prod_all)
            np.save(os.path.join(save_path, 'y_pred_all.npy'), y_pred_all)
            np.save(os.path.join(save_path, 'y_true_all.npy'), y_true_all)

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

    def metric_to_str(self, metrics):
        format_str = ''
        for k in metrics:
            if k == 'nloss':
                format_str += ' | {} = {:0.5f}'.format(k.upper(), metrics[k].mean())
            else:
                format_str += ' | {} = {:0.5f}'.format(k.upper(), metrics[k])
        return format_str

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


    def _stop_condition(self, epoch, patience=10):
        """
        Checks have not exceeded max epochs and has not gone patience epochs without improvement.
        """
        no_improvement = epoch >= self._best_epoch + patience
        exceeded_max_epochs = epoch >= self.config['max_epochs']
        return False if exceeded_max_epochs or no_improvement else True


    # Graph regularization
    def add_batch_graph_loss(self, out_adj, features, keep_batch_dim=False):
        # Graph regularization
        if keep_batch_dim:
            graph_loss = []
            for i in range(out_adj.shape[0]):
                L = torch.diagflat(torch.sum(out_adj[i], -1)) - out_adj[i]
                graph_loss.append(self.config['smoothness_ratio'] * torch.trace(torch.mm(features[i].transpose(-1, -2), torch.mm(L, features[i]))) / int(np.prod(out_adj.shape[1:])))

            graph_loss = to_cuda(torch.Tensor(graph_loss), self.device)

            ones_vec = to_cuda(torch.ones(out_adj.shape[:-1]), self.device)
            log_degrees = torch.log(torch.matmul(out_adj, ones_vec.unsqueeze(-1)) + Constants.VERY_SMALL_NUMBER)
            total_degrees = torch.matmul(ones_vec.unsqueeze(1), log_degrees).squeeze(-1).squeeze(-1)
            degree_regularization = -self.config['degree_ratio'] * total_degrees / out_adj.shape[-1]
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


def save_y_probs(a, b, path):
    y_true_path = path + '/y_true.npy'
    y_probs_path = path + '/y_probs.npy'
    np.save(y_true_path, a)
    np.save(y_probs_path, b)


def save_y_probs_patient(a, b, path):
    y_true_patient_path = path + '/y_true_patient.npy'
    y_probs_patient_path = path + '/y_probs_patient.npy'
    np.save(y_true_patient_path, a)
    np.save(y_probs_patient_path, b)

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

