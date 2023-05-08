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

import pandas as pd
import numpy as np
import alpaca
from alpaca_trade_api import REST
import matplotlib.pyplot as plt
import datetime
from tqdm import tqdm
import pytz

### get stock data
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.historical import StockHistoricalDataClient

import sys
sys.path = sys.path + ['../']

import my_secrets
import pickle

from warnings import filterwarnings
filterwarnings('ignore')

import time

stock_code='TQQQ'
INPUT_SEQUENCE_LENGTH = 1000
input_size = 1  # Number of features for each minute
hidden_size = 64  # Size of the hidden layer
output_size = 1  # Predicting one value: the minute-level price change
num_layers = 2  # Number of transformer encoder layers
dropout = 0.2  # Dropout probability
MIN_DELTA = 1e-8
LEARNING_RATE = 0.001
MIN_LR = 1e-8
NUM_EPOCHS = 2000
NHEAD=1

path = f'./historical_price_data/{stock_code}.csv'
data = pd.read_csv(path,index_col=0)
data.index = pd.Series([i.replace(tzinfo=None) for i in pd.to_datetime(data.index)]).dt.tz_localize(pytz.timezone('US/Eastern'))
a = data.resample('1min').mean().dropna()
a['change_rate'] = (a['close'] - a['open'])/a['open']
a['change_bi'] = np.where((a['close'] - a['open'])>0,1,0)
a['date'] = pd.to_datetime([i.date() for i in a.index])
a['year'] = a.date.dt.year
a['WOY'] = a.date.dt.week
a['WOY_year'] = [str(a)+str(b) for a,b in zip(a['WOY'],a['year'])]
a['MOY'] = a.date.dt.month
a['MOY_year'] = [str(a)+str(b) for a,b in zip(a['MOY'],a['year'])]
a['DOW'] = a.date.dt.day_of_week
a['DOY'] = a.date.dt.day_of_year
a['DOY_year'] = [str(a)+str(b) for a,b in zip(a['DOY'],a['year'])]
a['MOD'] = [i.hour*60+i.minute for i in a.index]
a['HOD'] = [i.hour for i in a.index]
a = a[(a.MOD>=9.5*60) & (a.MOD<=16*60)]
a['change_rate'] = (a['close'] - a['open'])/a['open']
whole_sequence = a['change_rate'].values
del a

###### loader
class myloader():
    def __init__(self, whole_sequence, past_feature_size=150000, batch_size=10):
        self.whole_sequence = whole_sequence
        self.past_feature_size = past_feature_size
        self.batch_size = batch_size
        self.total = int(len(whole_sequence)/batch_size)
    
    def get_data(self):
        batch_size = self.batch_size
        ## random idx
        
        idx_pool = list(range(len(self.whole_sequence)))[self.past_feature_size+10:]
        np.random.shuffle(idx_pool)
        
        x_list = []
        y_list = []
        for idx in idx_pool:
            if idx<=self.past_feature_size+10:
                continue
            else:
                
                x = np.expand_dims(
                    self.whole_sequence[idx-self.past_feature_size:idx].reshape(self.past_feature_size,-1),
                    axis=0
                )
                x = torch.tensor(x).float()*100
                y = torch.tensor(self.whole_sequence[idx]).reshape(1,-1).float()*100
                x_list.append(x)
                y_list.append(y)
                
                if not len(x_list)>=batch_size:
                    continue
                else:
                    x = torch.cat(x_list)
                    y = torch.cat(y_list)
                    x_list = []
                    y_list = []
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
    def __init__(self, optimizer, patience=5, min_lr=MIN_LR, factor=0.5):
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
        


######### model
class StockPricePredictionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.2):
        super(StockPricePredictionModel, self).__init__()
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(input_size, nhead=NHEAD, dim_feedforward=hidden_size, dropout=dropout, batch_first=True),
            num_layers
        )
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.transformer_encoder(x)
        x = self.fc1(x[:,-1,:])
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x

    
    
    

######## loader
train_loader = myloader(whole_sequence[0:int(len(whole_sequence)*0.6)],
                        INPUT_SEQUENCE_LENGTH,
                        batch_size=50
                       )
val_loader = myloader(whole_sequence[int(len(whole_sequence)*0.65):],
                        INPUT_SEQUENCE_LENGTH,
                      batch_size=50
                       )

######## model
model = StockPricePredictionModel(input_size, hidden_size, output_size, num_layers, dropout)
model = model.to('cpu').float()




# logging
wandb.init(
    project="TQQQ_prediciton_pure_hist_price",
    config={
    "learning_rate": LEARNING_RATE,
    "architecture": "Transformer",
    "dataset": "TQQQ",
    "epochs": NUM_EPOCHS,
    }
)

######## 
min_val_loss = np.inf
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
lr_scheduler = LRScheduler(optimizer)
early_stopping = EarlyStopping()
criterion = nn.MSELoss()

for epoch in range(NUM_EPOCHS):
    
    train_losses = []
    val_losses = []
    r2_list = []
    spearmanr_list = []
    
    #### train
    model.train()
    optimizer.zero_grad()
    for batch_count,(x,y) in tqdm(enumerate(train_loader.get_data()), total = train_loader.total):
        if batch_count>500:
            break
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        with torch.no_grad():
            train_losses.append(loss.item())
        optimizer.step()
    
    ### eval
    model.eval()
    for batch_count,(x,y) in tqdm(enumerate(val_loader.get_data()), total = val_loader.total):
        if batch_count>500:
            break
        preds = model(x)
        loss = criterion(preds, y)
        with torch.no_grad():
            val_losses.append(loss.item())
            
        ### r2 and spearman_r
        r2 = r2_score(y.detach().numpy().flatten(),
                 preds.detach().numpy().flatten()
                )
        spearman_r = spearmanr(y.detach().numpy().flatten(),
                 preds.detach().numpy().flatten()
                )[0]
        r2_list.append(r2)
        spearmanr_list.append(spearman_r)
    
    #### summary
    with torch.no_grad():
        train_epoch_loss = np.mean(train_losses) 
        val_epoch_loss = np.mean(val_losses)

        if val_epoch_loss < min_val_loss:
            min_val_loss = val_epoch_loss
            best_model = model
            with open('./best_transformer_for_TQQQ.pkl','wb') as f:
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
                   'val_r2':np.mean(r2_list), 'val_spearmanr':np.mean(spearmanr_list), 'lr':this_lr})

                
    print(f"Epoch {epoch+1}, train_loss: {train_epoch_loss}, val_loss: {val_epoch_loss}")
    
    
wandb.finish()

