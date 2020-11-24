## The Main file to train the neural network
import torch as t
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
#from torch.utils.tensorboard import SummaryWriter
from dataset import get_train_dataset, get_test_dataset
from trainer import Trainer
# from PadCollate import PadCollate

## load data
from merge_dataset import read_rawdata_to_person, zero_mean_one_person, zero_mean_person, data_split
## Models
from model.cnn1d import CnnNet
from model.cnn_rnn1d import Cnn_RnnNet
from model.cnn_lstm1d import Cnn_LstmNet
from model.Resnet_LstmNet import BasicBlock, ResNet_lstm


# current path and data path
cwd = os.getcwd()
data_folder = "../../data/data_stage2/STABILO_CHALLENGE_STAGE_2/"
plot_folder = "plot/cnn_lstm1d/final/"
data_path = os.path.join(cwd, data_folder)
plot_path = os.path.join(cwd, plot_folder)

# for hyper-parameters
alpha = 0.2
all_data = True
resample = True

batch_size = 16
learning_rate = 3e-5
weight_decay = 1e-3
norm = None
num_epochs = 350
cude_if = True

## Tensorboard
writer = None
#writer = SummaryWriter(log_dir='board_view/simple_endecoder/whole_data')

## Stage 2 Letter Data Set
persons = read_rawdata_to_person(data_path, all_data, millisec= 1000, resample=resample)
dataset = data_split(persons, alpha=alpha)
trainset = get_train_dataset(dataset)
testset = get_test_dataset(dataset)

### write own dataloader
trainloader = t.utils.data.DataLoader(trainset,
                                      batch_size= batch_size,
                                      shuffle= True)
testloader = t.utils.data.DataLoader(testset,
                                     batch_size= batch_size,
                                     shuffle= True)
## Model Setting
model = Cnn_LstmNet(10, 32)
# model = ResNet_lstm(BasicBlock, [2, 2, 1, 1])

## Loss function and Optimizer
criterion = t.nn.CrossEntropyLoss()
optimizer = t.optim.Adam(model.parameters(), lr= learning_rate, weight_decay= weight_decay)

early_stop = None

cnn_Trainer = Trainer(model, criterion, optimizer,
                     trainloader, testloader,
                     norm= norm,
                     cuda= cude_if,
                     early_stopping_cb= early_stop,
                     writer= writer) ## Tensorboard

train_losses, test_losses, train_accs, test_accs, f1s, df_cm = cnn_Trainer.fit(epochs= num_epochs)

## Plot the results
plt.figure()
plt.plot(np.arange(len(train_losses)), train_losses, label= 'train loss')
plt.plot(np.arange(len(test_losses)), test_losses, label= 'test loss')
plt.title('Loss')
plt.legend()
plt.savefig(plot_path + 'losses_ep350_lr3e-5_dropout0.2_weightdecay_1e-3_gyrorad_cnnlstm1d_32kernels_relu.png')

plt.figure()
plt.plot(np.arange(len(train_accs)), train_accs, label= 'Train accuracy')
plt.plot(np.arange(len(test_accs)), test_accs, label= 'Test accuracy')
plt.title('Accuracy')
plt.legend()
plt.savefig(plot_path + 'accuracy_ep350_lr3e-5_dropout0.2_weightdecay_1e-3_gyrorad_cnnlstm1d_32kernels_relu.png')

plt.figure(figsize=(24,16))
sn.heatmap(df_cm, annot=True)
plt.title('Confusion Matrix')
plt.savefig(plot_path + 'confusion_matrix_ep350_lr3e-5_dropout0.2_weightdecay_1e-3_gyrorad_cnnlstm1d_32kernels_relu.png')
