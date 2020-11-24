import numpy as np
import torch as t
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix
import sys, os

cwd = os.getcwd()

## Labels
LABELS = np.arange(52)
## Class
CLASSES = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11,
                'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22,
                'X': 23, 'Y': 24, 'Z': 25, 'a': 26, 'b': 27, 'c': 28, 'd': 29, 'e': 30, 'f': 31, 'g': 32, 'h': 33,
               'i': 34, 'j': 35, 'k': 36, 'l': 37, 'm': 38, 'n': 39, 'o': 40, 'p': 41, 'q': 42, 'r': 43, 's': 44,
               't': 45, 'u': 46, 'v': 47, 'w': 48, 'x': 49, 'y': 50, 'z': 51, '0': 53, '*': 52} ## '0' blank

KEY_LIST = list(CLASSES.keys())
VALUE_LIST = list(CLASSES.values())

## Device Configuration
device = t.device('cuda' if t.cuda.is_available() else 'cpu')

class Trainer:

    def __init__(self,
                 model,
                 loss_func,
                 optim= None, ## optimizer
                 train_set= None,
                 test_set= None,
                 norm = 'L1',
                 cuda= True,
                 early_stopping_cb= None,
                 writer= None):
        """
        Args:
            model: Model to be trained
            loss_func: Loss Function
            optim: Optimizer
            train_data: Training Dataset
            test_data: Testing Dataset
            norm: Regularization L1 or L2
            cuda: whether to use the GPU
            early_stopping_cb: The stopping criterion
            writer: Tensorboard
        """
        self._model = model
        self._loss_func = loss_func
        self._optim = optim
        self._trainset = train_set
        self._testset = test_set
        self._norm = norm
        self._cuda = cuda
        self._early_stopping = early_stopping_cb
        self._writer = writer
        if cuda:
            self._model = self._model.cuda()
            self._loss_func = loss_func.cuda()

    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, cwd + '/checkpoints/all_writers/cnn_lstm1d/lr_3e_5/gyrorad_dropout_weightdecay_32kernels_relu/checkpoint_{:03d}.ckp'.format(epoch))
    def restore_checkpoint(self, epoch_n):
        model_ckp = t.load(cwd + '/checkpoints/all_writers/cnn_lstm1d/lr_3e_5/gyrorad_dropout_weightdecay_32kernels_relu/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(model_ckp['state_dict'])

    def train_step(self, x, y):
        self._optim.zero_grad()
        predicted = self._model(x)

        loss = self._loss_func(predicted, y.to(device))
        lambda1 = 0.5
        if self._norm is 'L1':
            all_conv1_params = t.cat([x.view(-1) for x in self._model.conv1.parameters()])
            all_conv2_params = t.cat([x.view(-1) for x in self._model.conv2.parameters()])
            all_conv3_params = t.cat([x.view(-1) for x in self._model.conv3.parameters()])
            l1_regularization = lambda1 * (t.norm(all_conv1_params, 1)) # + t.norm(all_conv2_params, 1) + t.norm(all_conv3_params, 1))
            loss += l1_regularization
        loss.backward()
        self._optim.step()

        return loss.item(), predicted

    def train_epoch(self):
        self._model.train()
        train_losses = []
        correct = 0
        sum_labels = 0
        for i, (data, label) in enumerate(self._trainset, 0): ## data.size = [batch, length, features]
            data = data.float()   ## seq_len
            data = data.to(device)
            label = label.to(device)

            ## Forward pass
            loss, output = self.train_step(data, label)

            ## calculate the accuracy
            output = t.argmax(output, dim=1) ## compute the highest probability
            correct += (output == label).sum().item()
            sum_labels += len(label)

            train_losses.append(loss)
        avg_loss = np.mean(train_losses)
        avg_acc = correct / sum_labels * 100
        print('Train Avg. Loss: {}'.format(avg_loss))
        print('-------------------------------------------')
        print('Train avg character accuracy rate: {} %'.format(avg_acc))
        print('-------------------------------------------')
        return avg_loss, avg_acc

    def test_step(self, x, y):
        predicted = self._model(x)

        loss = self._loss_func(predicted, y.to(device))
        return loss, predicted

    def test_val(self):
        self._model.eval()
        test_loss = []
        ground_true = None
        predicted_label = None
        correct = 0
        sum_labels = 0
        with t.no_grad():
            for i, (data, label) in enumerate(self._testset, 0): ## seq_len
                data = data.float()
                data = data.to(device)
                label = label.to(device)

                loss, output = self.test_step(data, label)

                ## calculate the accuracy
                output = t.argmax(output, dim=1)  ## compute the highest probability
                correct += (output == label).sum().item()
                sum_labels += len(label)

                test_loss.append(loss)

                ## for f1 score ## collect all predict and label
                if predicted_label is None:
                    predicted_label = output
                else:
                    predicted_label = t.cat((predicted_label, output), dim=0)
                if ground_true is None:
                    ground_true = label
                else:
                    ground_true = t.cat((ground_true, label), dim=0)

            ## f1 score calculation
            f1 = f1_score(ground_true.cpu().numpy(), predicted_label.cpu().numpy(), average='macro')
            avg_loss = t.mean(t.stack(test_loss))
            avg_acc = 100 * correct / sum_labels
            print('Test Avg. Loss: {}'.format(avg_loss))
            print('-------------------------------------------')
            print('Test avg character accuracy rate: {} %'.format(avg_acc))
            print('-------------------------------------------')
            print('f1 score', f1)

        return avg_loss, avg_acc, f1

    def fit(self, epochs= -1):
        assert self._early_stopping is not None or epochs > 0
        train_losses = []
        train_accs = []
        test_losses = []
        test_accs = []
        f1s = []
        self.epoch_count = 1
        best_score = np.Inf
        best_acc = -np.Inf
        best_epoch = None
        best_epoch_acc = None
        while True:
            if self.epoch_count > epochs:
                print('best_epoch for loss: {}'.format(best_epoch))
                print('best_epoch for acc : {}'.format(best_epoch_acc))
                print('---- confusion matrix for the best epoch acc')
                self.restore_checkpoint(best_epoch_acc)
                self._model.eval()
                labels = None
                predictions = None
                with t.no_grad():
                    for i, (data, label) in enumerate(self._testset, 0):  ## seq_len
                        data = data.float()
                        data = data.to(device)
                        label = label.to(device)
                        predicted = self._model(data)
                        ## calculate the accuracy
                        predicted = t.argmax(predicted, dim=1)  ## compute the highest probability
                        if predictions is None:
                            predictions = predicted
                        else:
                            predictions = t.cat((predictions, predicted), dim=0)
                        if labels is None:
                            labels = label
                        else:
                            labels = t.cat((labels, label), dim=0)
                    predictions = np.asarray(predictions.cpu().numpy()).reshape(-1, 1)
                    labels = np.asarray(labels.cpu().numpy()).reshape(-1, 1)
                    cm = confusion_matrix(labels, predictions, labels=LABELS)
                    df_cm = pd.DataFrame(cm, index= [i for i in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"],
                                         columns=[i for i in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"])
                break
            print('epoch : {}'.format(self.epoch_count))
            train_loss, train_acc = self.train_epoch()
            test_loss, test_acc, f1 = self.test_val()

            if test_loss < best_score:
                best_epoch = self.epoch_count
                best_score = test_loss
                self.save_checkpoint(best_epoch)
            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch_acc = self.epoch_count
                self.save_checkpoint(self.epoch_count)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            f1s.append(f1)

            if self._early_stopping is not None:
                self._early_stopping.step(test_loss)
                self._early_stopping.should_stop()
                if self._early_stopping.early_stop is True:
                    print('The best score is on epoch {} and the accuracy {} %'.format(best_epoch, best_score))
                    break
            self.epoch_count += 1
        return train_losses, test_losses, train_accs, test_accs, f1s, df_cm