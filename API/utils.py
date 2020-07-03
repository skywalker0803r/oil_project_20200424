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

class transform_5423(object):
    def __init__(self):
        self.x_cols = y54.columns.tolist()
        self.y_cols = y23.columns.tolist()
        self.W = W
    
    def __call__(self,x):
        y23 = (x/self.W.sum(axis=0).values).fillna(0)@self.W.T.values
        y23.columns = self.y_cols
        return y23

class Dual_net(nn.Module):
    def __init__(self):
        super(Dual_net,self).__init__()
        
        # wt%中為0欄位
        self.zero_col = zero_col 
        
        #輕質部份
        self.c7_col = c7_col
        self.drop_c7_col = drop_c7_col
        
        #核心部份
        self.c6_col = c6_col 
        self.drop_c6_col = drop_c6_col
        
        #重質部份
        self.xhe_col = xhe_col
        
        #分離係數部份
        self.sp_zero_col = sp_zero_col 
        self.sp_one_col = sp_one_col
        
        # 輸入輸出設定
        C_in = 4
        C_out = 3 
        
        N_in = 54
        N_out = 54
        
        F_in = C_out + N_out
        F_out = C_out + N_out
        
        O_out = 3
        
        # 分離係數索引
        self.sle_idx = sle_idx
        self.shc_idx = shc_idx
        self.she_idx = she_idx
        
        # 建立網路
        self.C_net = self._build_C_net(C_in,C_out)
        self.N_net = self._build_N_net(N_in,N_out) 
        self.F_net = self._build_F_net(F_in,F_out)
        for i in range(54):
            setattr(self,'O_net{}'.format(i+1),self._build_O_net(F_out,O_out))
        
        # 初始化網路權重
        self.apply(self._init_weights)
            
    def forward(self,x):
        
        # 取得 case 和 xna
        case,xna = self._Fetch(x)
        
        # 取得 c6_total 和 c7_total
        self.c6_total = case[:,-1].reshape(-1,1)
        self.c7_total = case[:,1].reshape(-1,1)
        
        # case xna 分別 forward 後再 combine 到一起
        f = torch.cat((self.C_net(case),self.N_net(xna)),dim=1)
        
        # forward Onet 分離係數三個一組做預測
        output = torch.tensor([])
        for i in range(54):
            O_net = getattr(self,'O_net{}'.format(i+1))
            v = F.sigmoid(O_net(f)) # 區間縮放到[0,1]
            v = v / torch.sum(v,dim=1).reshape(-1,1) # 分離係數加總為1
            output = torch.cat((output,v),dim=1)    
        
        # 分離係數有部分可以強制為0或1
        output[:,self.sp_zero_col] = 0
        output[:,self.sp_one_col] = 1
        
        # 按照"層(輕核重)"將分離係數分三組
        sle = output[:,self.sle_idx]
        shc = output[:,self.shc_idx]
        she = output[:,self.she_idx]
        
        # 記錄分離係數 以便之後使用
        self.sle = sle
        self.shc = shc
        self.she = she
        
        # 重建組成 簡稱 wt%
        xle = self.reconstruction(xna,sle)
        xhc = self.reconstruction(xna,shc)
        xhe = self.reconstruction(xna,she)
        
        # combine 三層 wt%
        y = torch.cat((xle,xhc,xhe),dim=1)
        
        # wt%有部份成份可以強制為0
        y[:,self.zero_col] = 0
        
        # 質量平衡(輕質部份)
        y[:,self.c7_col] = self.normalize(y[:,self.c7_col]) * self.c7_total
        y[:,self.drop_c7_col] = self.normalize(y[:,self.drop_c7_col]) * (100 - self.c7_total)
        
        # 質量平衡(核心部份)
        y[:,self.c6_col] = self.normalize(y[:,self.c6_col]) * self.c6_total
        y[:,self.drop_c6_col] = self.normalize(y[:,self.drop_c6_col]) * (100 - self.c6_total)
        
        # 質量平衡(重質部份)
        y[:,self.xhe_col] = self.normalize(y[:,self.xhe_col]) * 100
        
        return y
    
    def normalize(self,x):
        return x / x.sum(dim=1).reshape(-1,1)
    
    def reconstruction(self,xna,s):
        return (100*xna*s)/torch.diag(xna@s.T).reshape(-1,1)
    
    @staticmethod
    def _Fetch(x):
        return x[:,:4],x[:,4:]
    
    @staticmethod
    def _build_C_net(input_shape,output_shape):
        net = torch.nn.Sequential(
            Linear(input_shape,128),
            Tanh(),
            Linear(128,output_shape))
        return net
    
    @staticmethod
    def _build_N_net(input_shape,output_shape):
        net = torch.nn.Sequential(
            Linear(input_shape,128),
            Tanh(),
            Linear(128,output_shape))
        return net
    
    @staticmethod
    def _build_F_net(input_shape,output_shape):
        net = torch.nn.Sequential(
            Linear(input_shape,128),
            Tanh(),
            Linear(128,output_shape))
        return net
    
    @staticmethod
    def _build_O_net(input_shape,output_shape):
        net = torch.nn.Sequential(
            Linear(input_shape,128),
            Tanh(),
            Linear(128,output_shape))
        return net
    
    @staticmethod
    def _init_weights(m):
        if hasattr(m,'weight'):
            torch.nn.init.xavier_uniform(m.weight)
        if hasattr(m,'bias'):
            m.bias.data.fill_(0)

class ANN_wrapper(object):
    def __init__(self,x_col,y_col,n_col,net):
        self.x_col = x_col
        self.y_col = y_col
        self.n_col = n_col
        self.net = net
        self.col_names = col_names
        self.s_col = s_col
    
    def predict(self,x):
        x = torch.tensor(x.values,dtype=torch.float)
        y = self.net(x).detach().cpu().numpy()
        y = pd.DataFrame(y,columns=self.y_col)
        assert np.all(y.values >= 0)
        
        sp_pred = np.hstack((self.net.sle.detach().numpy(),
                             self.net.shc.detach().numpy(),
                             self.net.she.detach().numpy()))
        sp_pred = pd.DataFrame(sp_pred,columns=self.col_names['sle']+self.col_names['shc']+self.col_names['she'])
        sp_pred = sp_pred[self.s_col]
        
        return y,sp_pred

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
        x = F.sigmoid(self.fc3(x))
        return x

class ANN_energy_wrapper(object):
    def __init__(self,x_col,y_col,mm_x,mm_y,net):
        self.x_col = x_col
        self.y_col = y_col
        self.mm_x = mm_x
        self.mm_y = mm_y
        self.net = net
    
    def predict(self,x):
        x = self.mm_x.transform(x)
        x = torch.tensor(x,dtype=torch.float)#.cuda()
        y = self.net(x).detach().cpu().numpy()
        y = self.mm_y.inverse_transform(y)
        y = pd.DataFrame(y,columns=self.y_col)
        assert np.all(y.values >= 0)
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

class model4333(nn.Module):
    def __init__(self,input_shape,output_shape):
        super(model4333,self).__init__()
        self.fc1 = Linear(input_shape,256,bias=False)
        self.fc2 = Linear(256,128,bias=False)
        self.fc3 = Linear(128,output_shape,bias=False)
        self.dropout = Dropout(0.1)
    
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.sigmoid(self.fc3(x))
        return x

class model_4333_wraper:
    def __init__(self):
        self.x_col = x_col
        self.y_col = y_col
        self.mm_x = mm_x
        self.mm_y = mm_y
        self.net = net.eval()
    
    def predict(self,x):
        feed = x.iloc[:,8:8+33].sum(axis=1).values.reshape(-1,1)
        x = self.mm_x.transform(x)
        x = torch.tensor(x,dtype=torch.float)#.cuda()
        y = self.net(x)
        y = y.detach().cpu().numpy()
        y = self.mm_y.inverse_transform(y)
        y = self.normalize(y,feed)
        y = pd.DataFrame(y,columns=self.y_col)
        output = y.sum(axis=1).values.reshape(-1,1)
        assert np.allclose(feed,output)
        assert np.all(y.values >= 0)
        return y
    
    def normalize(self,x,feed):
        x = x / x.sum(axis=1).reshape(-1,1)
        x *= feed
        return x

class model_tray(nn.Module):
    def __init__(self,input_shape,output_shape):
        super().__init__()
        self.fc1 = Linear(input_shape,128)
        self.fc2 = Linear(128,128)
        self.fc3 = Linear(128,output_shape)
    
    def forward(self,x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x

class model_tray_wrapper(object):
    def __init__(self,x_col,y_col,mm_x,mm_y,net):
        self.x_col = x_col
        self.y_col = y_col
        self.mm_x = mm_x
        self.mm_y = mm_y
        self.net = net
    
    def predict(self,x):
        x = self.mm_x.transform(x)
        x = torch.tensor(x,dtype=torch.float)#.cuda()
        y = self.net(x).detach().cpu().numpy()
        y = self.mm_y.inverse_transform(y)
        y = pd.DataFrame(y,columns=self.y_col)
        assert np.all(y.values >= 0)
        return y

class EVA(object):
    def __init__(self):
        self.A = joblib.load('./model/SVR(4_to_23).pkl')
        self.B = joblib.load('./model/transformer(23_to_54).pkl')
        self.C = joblib.load('./model/ANN(58_to_y162(二合一)).pkl')# 預測wt%同時給出分離係數
        #self.D = joblib.load('./model/transformer(SP162_to_Y162).pkl') #不需要了
        self.E = joblib.load('./model/ANN(energy).pkl')
        self.F = joblib.load('./model/transformer(54_to_33).pkl')
        self.G = joblib.load('./model/transformer(33_to_15).pkl')
        self.H = joblib.load('./model/transformer(43_to_33).pkl')
        self.I = joblib.load('./model/transformer(54_to_23).pkl')
        
        self.col_names = joblib.load('./data/phase_2/cleaned/col_names.pkl')

    def __call__(self, X):
        x4, case4 = X.iloc[:, :4], X.iloc[:, 4:]
        # model forward
        self.y23 = self.A.predict(x4)  # percent sum = 100
        self.Xna = self.B(self.y23)  # percent sum = 100
        # percent sum = 1 * 54component = 54
        self.y162,self.sp162 = self.C.predict(case4.join(self.Xna))
        
        self.xhc = self.y162[self.col_names['xhc']]  # percent sum = 100
        self.xhc33_p = self.F(self.xhc)/100  # percent

        # predict duty and density(ton/m3)
        E_pred = self.E.predict(case4.join(self.Xna))
        self.duty = E_pred.iloc[:, :2]
        self.density = E_pred.iloc[:, 2:]

        # get split factor
        self.shc = self.sp162[self.col_names['shc']]
        self.sle = self.sp162[self.col_names['sle']]
        self.she = self.sp162[self.col_names['she']]

        # fhc_ton = fhc_m3*density
        self.fhc_m3 = case4['Case Conditions_Heart Cut Prod. Rate (Input)_m3/hr']
        self.fhc_ton = self.fhc_m3.values * self.density.iloc[:, 2].values

        # calculate fle_ton,fhe_ton
        self.fle_ton = self.fhc_ton * ((self.Xna.values@self.sle.values.ravel()) /
                                       (self.Xna.values@self.shc.values.ravel()))
        self.fhe_ton = self.fhc_ton * ((self.Xna.values@self.she.values.ravel()) /
                                       (self.Xna.values@self.shc.values.ravel()))

        # calculate fna
        self.fna_ton = self.fle_ton + self.fhc_ton + self.fhe_ton

        # scale fle,fhc,fhe
        self.fle_ton = self.fle_ton*100000/self.fna_ton
        self.fhc_ton = self.fhc_ton*100000/self.fna_ton
        self.fhe_ton = self.fhe_ton*100000/self.fna_ton

        # add columns name
        self.fle_ton = pd.Series(
            self.fle_ton, name=self.col_names['Rate_ton'][1])
        self.fhe_ton = pd.Series(
            self.fhe_ton, name=self.col_names['Rate_ton'][3])

        # scale duty
        self.duty *= 100/self.fna_ton[0]

        # xhc33 = fhc_ton*xhc_33_p
        self.xhc33 = self.fhc_ton[0]*self.xhc33_p
        self.xhc33_tmp = self.xhc33*74000/self.fhc_ton[0]

        # xhe33 = H(someting + xhc33 + something)
        xhc23 = self.I(self.xhc)
        C6Pm = xhc23[['C5NP', 'C5IP', 'C6IP', 'C6NP']].sum(axis=1)[0]

        NpA = xhc23[['C5N', 'C6N', 'C7N', 'C8N', 'C9N', 'C10N']].sum(axis=1)[0] +\
            xhc23[['C6A', 'C7A', 'C8A', 'C9A', 'C10A']].sum(axis=1)[0]*2

        self.H_input = [NpA, 0.942, 3.78, C6Pm, 517, 517, 517, 517] + \
            self.xhc33_tmp.values.ravel().tolist()+[4.798, 1.44]
        self.H_input = np.array([self.H_input])
        self.H_input = pd.DataFrame(self.H_input, columns=self.H.x_col)
        self.reformer_input = [NpA, 0.942, 3.78, C6Pm, 517, 517, 517, 517] + self.xhc33.values.ravel().tolist() + [4.798, 1.44]
        self.reformer_input = np.array([self.reformer_input])
        self.reformer_input = pd.DataFrame(self.reformer_input, columns=self.H.x_col)

        self.重組33_tmp = self.H.predict(self.H_input)

        self.重組33 = self.重組33_tmp *self.fhc_ton[0]/74000

        # final output
        self.重組15 = self.G(self.重組33)

        # raname and combine return to user
        self.predict = self.重組15.join(self.fle_ton).join(
            self.fhe_ton).join(self.duty)
        self.naphtha = self.y23
        self.pre_d = self.sp162
        self.reform = self.重組33
        
class Price_model:
    def __init__(self,各產品單價,回送輕油單價,輕油單價):
        self.各產品單價 = 各產品單價
        self.回送輕油單價 = 回送輕油單價
        self.輕油單價 = 輕油單價
    
    def compute_price(self,各重組油組成,產出量,回送輕油流量,輕油用量):
        total = 0
        total += (各重組油組成*產出量*self.各產品單價).sum() # (vector*scalar*vector).sum()
        total += 回送輕油流量*self.回送輕油單價 # scalar*scalar
        total -= 輕油用量*self.輕油單價 # scalar*scalar
        return total

class OperationOptimModel:
    def __init__(self,F,Q,case_list):
        self.case_list = case_list
        self.F = F
        self.Q = Q
    
    def get_advice(self,xna):
        fitness = []
        actions = []
        
        #對各種case遍歷取得a,h,r代入Q函數計算產值,返回最大產值對應的action
        print('產值計算中請稍等')
        for case in tqdm(self.case_list):
            
            # 使用操作建議模組取得 加熱建議a , 產物組成h , 流量r(入料量,回流量,輕流量,重流量)
            加熱量,產物,流量,加工成本 = self.F(case,xna)
            回送流量 = 流量.iloc[:,-2:].sum(axis=1).values[0]
            輕油用量 = 流量.iloc[:,0].values[0]
            產物 = 產物.values
            加工成本 = 加工成本.values
            actions.append(加熱量)
            
            #代入估價模組計算產值
            產值 = self.Q(產物,回送流量,輕油用量,加工成本)
            fitness.append(產值)
        
        # 選擇效益最大的產值對應的操作
        best_idx = np.array(fitness).argmax()
        return actions[best_idx],fitness[best_idx],fitness