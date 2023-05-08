import pandas as pd
import numpy as np
import os, sys, pytz
import matplotlib.pyplot as plt
from tqdm import tqdm
from warnings import filterwarnings
filterwarnings('ignore')

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
import wandb
import random
import pickle



### some hyperparameters
MIN_DELTA = 1e-8
MIN_LR = 1e-8
HIDDEN_SIZE = 64
NUM_LAYERS = 2
LEARNING_RATE = 0.001
NUM_EPOCHS = 200



### define data loader
class LargeDataset(Dataset):
    def __init__(self, data_dir, stock_code, block_index_list):
        self.stock_code = stock_code
        self.data_dir = data_dir
        self.file_list = [i for i in sorted(os.listdir(data_dir)) if stock_code in i and int(i.split('.pkl')[0].split('_')[-1]) in block_index_list]  # List all files in the data directory

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        x, y = np.expand_dims(data.iloc[:,1:].values, -1), data.iloc[:,0].values  # Assuming the data is stored as dictionaries with 'x' and 'y' keys
        return x, y

### define data loader
class subDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        x, y = self.x[idx,:], self.y[idx]  # Assuming the data is stored as dictionaries with 'x' and 'y' keys
        return x, y
    
class myloader():
    def __init__(self, data_dir, stock_code, block_index_list):
        self.stock_code = stock_code
        self.block_index_list = block_index_list
        self.overall_train_dataset = LargeDataset(data_dir,stock_code,block_index_list)
        self.current_sub_dataset = None
    
    def get_data(self):
        for block_idx in range(len(self.block_index_list)):
            X_block, y_block = self.overall_train_dataset[block_idx]
            sub_dataloader = DataLoader(
                    subDataset(X_block, y_block),
                    batch_size=1024,
                    shuffle=False
                )
            for x,y in sub_dataloader:
                yield x,y

                    



### define some utils
class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after patience epochs.
    """
    def __init__(self, patience=20, min_delta=MIN_DELTA):
        """
        :param patience: how many epochs to wait before stopping when loss is not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            # print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                # print('INFO: Early stopping')
                self.early_stop = True
                
                
class LRScheduler():
    """
    Learning rate scheduler. If the validation loss does not decrease for the 
    given number of `patience` epochs, then the learning rate will decrease by
    by given `factor`.
    """
    def __init__(self, optimizer, patience=5, min_lr=MIN_LR, factor=0.3):
        """
        new_lr = old_lr * factor
        :param optimizer: the optimizer we are using
        :param patience: how many epochs to wait before updating the lr
        :param min_lr: least lr value to reduce to while updating
        :param factor: factor by which the lr should be updated
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( 
                self.optimizer,
                mode='min',
                patience=self.patience,
                factor=self.factor,
                min_lr=self.min_lr,
                verbose=True
            )
    def __call__(self, val_loss):
        # take one step of the learning rate scheduler while providing the validation loss as the argument
        self.lr_scheduler.step(val_loss)
        
        
#### define the model
class LSTM_model(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super(LSTM_model, self).__init__()
        self.hidden_size = hidden_size # number of features in hidden state
        self.num_layers = num_layers #number of lstm layers
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.conv1 = nn.Conv1d(1, 8, kernel_size=64) # in_channels, out_channels, kernel_size
        self.pool1 = nn.MaxPool1d(kernel_size=4)
        self.conv2 = nn.Conv1d(8, 16,kernel_size=32)
        self.pool2 = nn.MaxPool1d(kernel_size=4)        
        self.conv3 = nn.Conv1d(16, 32,kernel_size=16)
        self.pool3 = nn.MaxPool1d(kernel_size=4)
        
        
        self.lstm_month = nn.LSTM(input_size=15, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, bidirectional=False) #lstm
        self.lstm_week = nn.LSTM(input_size=15, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, bidirectional=False) #lstm
        self.lstm_day = nn.LSTM(input_size=30, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, bidirectional=False) #lstm
        self.lstm_minute = nn.LSTM(input_size=84, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, bidirectional=False) #lstm

        
        self.fc_1 =  nn.Linear(266,256) #fully connected 1
        self.drop1 = nn.Dropout(p=0.1, inplace=False)
        self.fc_2 =  nn.Linear(256,32) #fully connected 2
        self.drop2 = nn.Dropout(p=0.1, inplace=False)
        self.fc_3 =  nn.Linear(32,1) #fully connected 3

        torch.nn.init.xavier_uniform_(self.fc_1.weight)
        torch.nn.init.xavier_uniform_(self.fc_2.weight)
        torch.nn.init.xavier_uniform_(self.fc_3.weight)
        
    def forward(self,x):
        ##### LSTM for month data
        month_data = x[:,:,0:15]
        h_0_month = Variable(torch.zeros(self.num_layers, month_data.size(0), self.hidden_size)).to(self.device)
        c_0_month = Variable(torch.zeros(self.num_layers, month_data.size(0), self.hidden_size)).to(self.device)
        output_month, (hn_month, cn_month) = self.lstm_month(month_data, (h_0_month, c_0_month)) # hn.shape = 1, N, 64
        hn_month = hn_month[hn_month.shape[0]-1:]
        hn_month = hn_month.view(-1, self.hidden_size) # hn.shape = N, 64
        
        ##### LSTM for week data
        week_data = x[:,:,15:30]
        h_0_week = Variable(torch.zeros(self.num_layers, week_data.size(0), self.hidden_size)).to(self.device)
        c_0_week = Variable(torch.zeros(self.num_layers, week_data.size(0), self.hidden_size)).to(self.device)
        output_week, (hn_week, cn_week) = self.lstm_month(week_data, (h_0_week, c_0_week)) # hn.shape = 1, N, 64
        hn_week = hn_week[hn_week.shape[0]-1:]
        hn_week = hn_month.view(-1, self.hidden_size) # hn.shape = N, 64
        
        ##### LSTM for day data
        day_data = x[:,:,30:60]
        h_0_day = Variable(torch.zeros(self.num_layers, day_data.size(0), self.hidden_size)).to(self.device)
        c_0_day = Variable(torch.zeros(self.num_layers, day_data.size(0), self.hidden_size)).to(self.device)
        output_day, (hn_day, cn_day) = self.lstm_day(day_data, (h_0_day, c_0_day)) # hn.shape = 1, N, 64
        hn_day = hn_day[hn_day.shape[0]-1:]
        hn_day = hn_day.view(-1, self.hidden_size) # hn.shape = N, 64
        
        
        ##### convolution for minute data
        # print(x.shape) # N, 1, 24
        out  = self.conv1(x[:,:,60:]) # default activation is None; out.shape: N, 8, 24
        out = self.pool1(out) # out.shape: N, 8, 12
        
        out = self.conv2(out) # out.shape: N, 16, 12
        out = self.pool2(out) # out.shape: N, 16, 6
        
        out = self.conv3(out) # out.shape: N, 32, 6
        out = self.pool3(out) # out.shape: N, 32, 3
        
        h_0 = Variable(torch.zeros(self.num_layers, out.size(0), self.hidden_size)).to(self.device)
        c_0 = Variable(torch.zeros(self.num_layers, out.size(0), self.hidden_size)).to(self.device)

        output, (hn, cn) = self.lstm_minute(out, (h_0, c_0)) # hn.shape = 1, N, 64
        hn = hn[hn.shape[0]-1]
        hn = hn.view(-1, self.hidden_size) # hn.shape = N, 64
        
        input_to_fcl = torch.cat((hn_month, hn_week, hn_day, hn, x[:,0,-10:]), axis=1)
        out = self.fc_1(input_to_fcl) # Dense, out.shape = N, 256
        out = self.drop1(out)
        out = self.fc_2(out) # out.shape = N, 32
        out = self.drop2(out)
        out = self.fc_3(out) # out.shape = N, 12
        return out
    
    
    
#### load data
train_loader = myloader('./training_data/','TQQQ',list(range(0,65)))
val_loader = myloader('./training_data/','TQQQ',list(range(65,75)))


#### define the model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if os.path.exists('./best_lstm_for_TQQQ.pkl'):
    lstm_model = pickle.load(open('best_lstm_for_TQQQ.pkl','rb'))
    print('model loaded')
else:
    lstm_model = LSTM_model(HIDDEN_SIZE, NUM_LAYERS)
    print('make new model')
    
lstm_model = lstm_model.to(device).float()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=LEARNING_RATE)
lr_scheduler = LRScheduler(optimizer)
early_stopping = EarlyStopping()
        
    
##### start training
min_val_loss = np.inf


# logging
wandb.init(
    project="TQQQ_prediciton_pure_hist_price",
    config={
    "learning_rate": LEARNING_RATE,
    "architecture": "LSTM-15month-15week-30day-5850min-10min",
    "dataset": "TQQQ",
    "epochs": NUM_EPOCHS,
    }
)

###### start
for epoch in range(NUM_EPOCHS):
    val_losses = []
    train_losses = []

    ## train
    lstm_model.train()
    for inputs, targets in tqdm(train_loader.get_data(),total=len(train_loader.block_index_list)*5):
        inputs = inputs.float()
        targets = targets.float()
        inputs = inputs.permute([0,2,1])
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        ### train, pred
        preds = lstm_model(inputs)
        loss = criterion(preds, targets)
        with torch.no_grad():
            train_losses.append(loss.item())
            
        ### back prop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    
    ### eval
    r2_list = []
    spearmanr_list = []
    lstm_model.eval()
    for inputs, targets in val_loader.get_data():
        inputs = inputs.float()
        targets = targets.float()
        inputs = inputs.permute([0,2,1])
        inputs = inputs.to(device)
        preds = lstm_model(inputs)
        val_loss = criterion(preds, targets)

        val_losses.append(val_loss.item())
        
        ### r2 and spearman_r
        r2 = r2_score(targets.detach().numpy().flatten(),
                 preds.detach().numpy().flatten()
                )
        spearman_r = spearmanr(targets.detach().numpy().flatten(),
                 preds.detach().numpy().flatten()
                )[0]
        r2_list.append(r2)
        spearmanr_list.append(spearman_r)

    with torch.no_grad():
        val_epoch_loss = np.mean(val_losses)
        train_epoch_loss = np.mean(train_losses)
        if epoch%1 == 0 or epoch==(NUM_EPOCHS-1):
            print("Epoch: {}/{}...".format(epoch, NUM_EPOCHS),
                  "Loss: {:.6f}...".format(train_epoch_loss),
                  "Val Loss: {:.6f}".format(val_epoch_loss))

        if val_epoch_loss < min_val_loss:
            min_val_loss = val_epoch_loss
#             best_hidden_size = hidden_size
            best_model = lstm_model
            with open('./best_lstm_for_TQQQ.pkl','wb') as f:
                pickle.dump(best_model, f) #### write model
    
        lr_scheduler(val_epoch_loss)
        early_stopping(val_epoch_loss)
        if early_stopping.early_stop:
            print("Early stopping after epoch: {}/{}...".format(epoch, NUM_EPOCHS),
                  "Loss: {:.6f}...".format(train_epoch_loss),
                  "Val Loss: {:.6f}".format(val_epoch_loss))            
            break
            
    for param_group in lr_scheduler.optimizer.param_groups:
        this_lr = param_group['lr']
    wandb.log({"train_loss": train_epoch_loss, "val_loss": val_epoch_loss, 
               'val_r2':np.nanmean(r2_list), 'val_spearmanr':np.nanmean(spearmanr_list), 'lr':this_lr})

    
wandb.finish()
    
    
    
