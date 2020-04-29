import pandas as pd
from sklearn.svm import SVR
from tqdm import tqdm_notebook as tqdm
import numpy as np
from sklearn.metrics import r2_score,mean_squared_error
from math import sqrt
from sklearn.preprocessing import StandardScaler
import warnings;warnings.simplefilter('ignore')
from sklearn.pipeline import Pipeline
import joblib
import pickle
import torch
import torchviz
from torch import nn
import torch.nn.functional as F
from torch import tensor
from torch.nn import Linear,ReLU,Sigmoid,Tanh
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import r2_score,mean_squared_error
from math import sqrt
import joblib
from torch.utils.tensorboard import SummaryWriter
import warnings;warnings.simplefilter('ignore')
from tqdm import tqdm_notebook as tqdm
import os
from sklearn.utils import shuffle

class custom_svr(object):
  def __init__(self,x_cols,y_cols):
    self.x_cols = x_cols
    self.y_cols = y_cols
    self.N_col = ['C5N','C6N','C6A','C7N','C7A','C8N','C8A','C9N','C9A','C10N','C10A']
    self.P_col = ['C5NP','C5IP','C6NP','C6IP','C7NP','C7IP','C8NP','C8IP','C9NP','C9IP','C10NP','C10IP']
    self.model_23 = {}
    for y_name in y_cols:
      self.model_23[y_name] = Pipeline([('scaler',StandardScaler()),('reg',SVR(C=0.3))])
  
  def fit(self,X,y):
    for y_name in tqdm(self.y_cols):
      self.model_23[y_name].fit(X,y[y_name])
      y_pred = self.model_23[y_name].predict(X) 
      # Sequence prediction add y_pred to X 
      X.loc[:,y_name] = y_pred
    # recover X
    X = X[self.x_cols]
  
  def predict(self,data):
    X = data.copy()    
    results = pd.DataFrame(index=[*range(len(X))],columns=self.y_cols)
    for y_name in self.y_cols:
      y_pred = self.model_23[y_name].predict(X)
      results.loc[:,y_name] = y_pred
      # Sequence prediction add y_pred to X 
      X.loc[:,y_name] = y_pred
    # recover X
    X = X[self.x_cols]
    
    # normalize depand on N+A and P
    X['P'] = 100 - X['N+A']
    results[self.N_col] = self._normalize(results[self.N_col])*X['N+A'].values.reshape(-1,1)
    results[self.P_col] = self._normalize(results[self.P_col])*X['P'].values.reshape(-1,1)

    return results
  
  @staticmethod
  def _normalize(x):
    return x/x.sum(axis=1).values.reshape(-1,1)

class transform_2354(object):
    def __init__(self):
        self.x_cols = y23.columns.tolist()
        self.y_cols = y54.columns.tolist()
        self.W = W
    
    def __call__(self,x):
        res = x.values@self.W
        return pd.DataFrame(res,columns=self.y_cols)

class Dual_net(nn.Module):
    def __init__(self):
        super(Dual_net,self).__init__()
        C_in = 4
        C_out = 3
        N_in = 54
        N_out = 54
        F_in = C_out+N_out
        F_out = C_out+N_out
        O_out = 3
        
        # build C,N,F
        self.C_net = self._build_C_net(C_in,C_out)
        self.N_net = self._build_N_net(N_in,N_out) 
        self.F_net = self._build_F_net(F_in,F_out)
        
        # build O_net
        for i in range(54):
            setattr(self,'O_net{}'.format(i+1),self._build_O_net(F_out,O_out))
        
        # initialize weight
        self.apply(self._init_weights)
            
    def forward(self,x):
        c,n = self._Fetch(x)
        c,n = self.C_net(c),self.N_net(n)
        f = torch.cat((c,n),dim=1)
        f = self.F_net(f)
        output = torch.tensor([]).cuda()
        for i in range(54):
            O_net = getattr(self,'O_net{}'.format(i+1))
            v = F.sigmoid(O_net(f))
            output = torch.cat((output,v),dim=1)
        return output
    
    @staticmethod
    def _Fetch(x):
        return x[:,:4],x[:,4:]
    
    @staticmethod
    def _build_C_net(input_shape,output_shape):
        net = torch.nn.Sequential(
            Linear(input_shape,128),
            Tanh(),
            Linear(128,output_shape))
        return net.cuda()
    
    @staticmethod
    def _build_N_net(input_shape,output_shape):
        net = torch.nn.Sequential(
            Linear(input_shape,128),
            Tanh(),
            Linear(128,output_shape))
        return net.cuda()
    
    @staticmethod
    def _build_F_net(input_shape,output_shape):
        net = torch.nn.Sequential(
            Linear(input_shape,128),
            Tanh(),
            Linear(128,output_shape))
        return net.cuda()
    
    @staticmethod
    def _build_O_net(input_shape,output_shape):
        net = torch.nn.Sequential(
            Linear(input_shape,128),
            Tanh(),
            Linear(128,output_shape))
        return net.cuda()
    
    @staticmethod
    def _init_weights(m):
        if hasattr(m,'weight'):
            torch.nn.init.xavier_uniform(m.weight)
        if hasattr(m,'bias'):
            m.bias.data.fill_(0)


class ANN_wrapper(object):
    def __init__(self,x_col,y_col,n_col,scaler,net):
        self.x_col = x_col
        self.y_col = y_col
        self.n_col = n_col
        self.scaler = scaler
        self.net = net
    
    def predict(self,x):
        x = self.scaler.transform(x)
        x = torch.tensor(x,dtype=torch.float).cuda()
        y = self.net(x).detach().cpu().numpy()
        y = pd.DataFrame(y,columns=self.y_col).apply(lambda x:round(x,12))
        y = self.normalize(y)
        return y
    
    def normalize(self,y):
        for i in range(0,162+1-3,3):
            col3 = self.y_col[i:i+3]
            assert len(col3) == 3
            y[col3] = y[col3].values / y[col3].sum(axis=1).values.reshape(-1,1)
        return y

class transformer162162(object):
    def __init__(self):
        # output columns
        self.le = col_names['xle']
        self.hc = col_names['xhc']
        self.he = col_names['xhe']
        
        # split factor columns
        self.le_sp = col_names['sle']
        self.hc_sp = col_names['shc']
        self.he_sp = col_names['she']
    
    @staticmethod
    def _calculate_output(X,S,col_name):
        X, S = X.values, S.values
        F = np.diag(X@(S.T)).reshape(-1,1)
        Y = 100*(X*S)/(F)
        return pd.DataFrame(Y,columns=col_name)
    
    def __call__(self,xna,sp162):
        sle = sp162[self.le_sp] #SLE
        shc = sp162[self.hc_sp] #SHC
        she = sp162[self.he_sp] #SHE
        x_le = self._calculate_output(xna,sle,self.le) #XLE
        x_hc = self._calculate_output(xna,shc,self.hc) #XHC
        x_he = self._calculate_output(xna,she,self.he) #XHE
        return pd.concat([x_le,x_hc,x_he],axis=1)

class energy_net(nn.Module):
    def __init__(self,input_shape,output_shape):
        super(energy_net,self).__init__()
        self.fc1 = Linear(input_shape,128)
        self.fc2 = Linear(128,128)
        self.fc3 = Linear(128,output_shape)
    
    def forward(self,x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

class ANN_energy_wrapper(object):
    def __init__(self,x_col,y_col,scaler,net):
        self.x_col = x_col
        self.y_col = y_col
        self.scaler = scaler
        self.net = net
    
    def predict(self,x):
        x = self.scaler.transform(x)
        x = torch.tensor(x,dtype=torch.float).cuda()
        y = self.net(x).detach().cpu().numpy()
        y = pd.DataFrame(y,columns=self.y_col)
        return y

class transformer_5433(object):
    def __init__(self,x_col,y_col,W):
        self.x_col = x_col
        self.y_col = y_col
        self.W = W
    
    def __call__(self,X):
        return pd.DataFrame(X.values @ self.W,columns=self.y_col)

class transformer_3315(object):
    def __init__(self,x_col,y_col,W):
        self.x_col = x_col
        self.y_col = y_col
        self.W = W
    def __call__(self,X):
        return pd.DataFrame(X.values@self.W,columns=self.y_col)

class EVA(object):
    def __init__(self):
        self.A = joblib.load('./model/SVR(4_to_23).pkl')
        self.B = joblib.load('./model/transformer(23_to_54).pkl')
        self.C = joblib.load('./model/ANN(58_to_sp162).pkl')
        self.D = joblib.load('./model/transformer(SP162_to_Y162).pkl')
        self.E = joblib.load('./model/ANN(energy).pkl')
        self.F = joblib.load('./model/transformer(54_to_33).pkl')
        self.G = joblib.load('./model/transformer(33_to_15).pkl')
        self.col_names = joblib.load('./data/phase_2/cleaned/col_names.pkl')
    
    def __call__(self,X):
        x4,case4 = X.iloc[:,:4],X.iloc[:,4:]
        self.y23 = self.A.predict(x4)
        self.y54 = self.B(self.y23)
        self.sp162 = self.C.predict(case4.join(self.y54))
        self.y162 = self.D(self.y54,self.sp162)
        self.hc54 = self.y162[self.col_names['xhc']]
        self.y33 = self.F(self.hc54)
        self.y15 = self.G(self.y33)
        #===========================
        E_pred = self.E.predict(case4.join(self.y54))
        self.duty = E_pred.iloc[:,:2]
        self.density = E_pred.iloc[:,2:]
        #===========================
        self.shc = self.sp162[self.col_names['shc']]
        self.sle = self.sp162[self.col_names['sle']]
        self.she = self.sp162[self.col_names['she']]
        #===========================
        self.fhc_m3 = case4['Case Conditions_Heart Cut Prod. Rate (Input)_m3/hr']
        self.fhc_ton = self.fhc_m3.values * self.density.iloc[:,2].values
        self.fle_ton = self.fhc_ton * ((self.y54.values@self.sle.values.ravel())/(self.y54.values@self.shc.values.ravel()))
        self.fhe_ton = self.fhc_ton * ((self.y54.values@self.she.values.ravel())/(self.y54.values@self.shc.values.ravel()))
        #========================================================
        self.fna_ton = self.fle_ton + self.fhc_ton + self.fhe_ton
        self.fna_ton_p = 200000
        self.fle_ton_p = 200000 * self.fle_ton/self.fna_ton
        self.fhc_ton_p = 200000 * self.fhc_ton/self.fna_ton
        self.fhe_ton_p = 200000 * self.fhe_ton/self.fna_ton
        self.fle_ton_p = pd.Series(self.fle_ton_p,name=self.col_names['Rate_ton'][1])
        self.fhe_ton_p = pd.Series(self.fhe_ton_p,name=self.col_names['Rate_ton'][3])
        #===========================
        self.duty *= 200/self.fna_ton[0]
        self.predict = self.y15.join(self.fle_ton_p).join(self.fhe_ton_p).join(self.duty)
        self.naphtha = self.y23
        self.pre_d = self.sp162
        self.reform = self.y33

class model4333(object):
    def __init__(self,x_col,y_col):
        self.x_col = x_col
        self.y_col = y_col
        self.model = {}
        for y_name in self.y_col:
            self.model[y_name] = LinearRegression()
            
    def fit(self,X,y):
        for name in tqdm(self.y_col):
            self.model[name].fit(X,y[name])
        print('fit done')
    
    def predict(self,X):
        result = pd.DataFrame(index=X.index,columns=self.y_col)
        for name in self.y_col:
            result[name] = self.model[name].predict(X)
        return result