"""
EEG-based Reading Skill Evaluation Model Design with Deep Learning Architecture
- Binary Classification based on the threshold
- Input: Zero-padded EEG data
- Target: Encoded score error data (0 if <thereshold, 1 if >=threshold)
- Subject Independent test with Leave-One-Out Cross Validation
Authors: Jihyeok Kim dkdnfk314@naver.com
Main Function (running on Spyder)

"""
#%%1
import torch

import argparse
import os
import numpy as np

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import mat73

from torch.utils.data import DataLoader

import time
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Parameters for model")
parser.add_argument("--experiment_name", type=str,
                    default="RSE_EEG_S1_Classification_Model_With_Padding", help="name of model")
parser.add_argument("--mat_dir", type=str,
                    default="./preprocessed_matfile/", help="directory of matfiles")
parser.add_argument("--n_ch_eeg", type=int,
                    default=32, help="the number of EEG channels")
parser.add_argument("--n_ch_eog", type=int,
                    default=5, help="the number of EOG channels")
parser.add_argument("--n_filters", type=int,
                    default=40, help="the number of filters in Shallownet")
parser.add_argument("--input_time_length", type=int,
                    default=512, help="the window size")
parser.add_argument("--esp", type=int,
                    default=5, help="the patience of early stopping")
parser.add_argument("--n_epochs", type=int,
                    default=100, help="the maximum number of epochs")
parser.add_argument("--cp_name", type=str,
                    default='checkpointEEGS1_binary_classification_p.pt', help="the filename of checkpoint")
parser.add_argument("--threshold", type=float,
                    default=0.0, help="setting the classification threshold")



args, unknown = parser.parse_known_args()
print(args)

#%%2
# gpu 사용
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

print(device)
#%%3
# load preprocessed matfile
# you should make "preprocessed matfile" folder and locate EEG (and EOG) and score data file there
file_list = os.listdir(args.mat_dir)
load_list = []
for file in file_list:
    filename = args.mat_dir + file
    data = mat73.loadmat(filename)
    load_list.append(data)
    
#%%4
# x data
s1_eeg = load_list[0]['eeg_s1_dataset']

#%%5
# label data
s1_score_err = load_list[5]['score1_err'].tolist()

#%%6-1
# visualizing score distribution
flatten_s1_score = [x for y in s1_score_err for x in y]
unique_value = set(flatten_s1_score)
xlabel = sorted(list(unique_value))
ylabel = [0]*(len(xlabel))

for i in flatten_s1_score:
    ylabel[xlabel.index(i)] += 1
        
plt.hist(flatten_s1_score, bins=64)
plt.title("Session 1 Score Distribution")
plt.xlabel("score value")
plt.ylabel("number")
plt.show()

# counting score based on threshold
threshold = args.threshold
count11, count12 = 0, 0
for i in flatten_s1_score:
    if i < threshold:
        count11 += 1
    else:
        count12 += 1
        
#%%6-2
# Generate encoded score class data
encoded_score1, encoded_score2 = [], []
for i in s1_score_err:
    tmp = []
    for j in i:
        if j < threshold:
            tmp.append(0)
        else:
            tmp.append(1)
    encoded_score1.append(tmp)

#%% - Padding (S1 EEG)
n_ch = args.n_ch_eeg
n_wl = 512

max_length = 0
for i in s1_eeg:
    for j in i:
        if j.shape[0] >= max_length:
            max_length = j.shape[0]

for idx1, i in enumerate(s1_eeg):
    for idx2, j in enumerate(i):
        if j.shape[0] < max_length:
            n = max_length - j.shape[0]
            j = np.concatenate([np.zeros((n, n_ch, n_wl), dtype=np.float32), j])
            s1_eeg[idx1][idx2] = j
            
#%%7
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.x_data = xdata    
        self.y_data = ydata
        
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        x = self.x_data[idx]     # return x data (sequence length, 32, 512)
        y = self.y_data[idx]    # return y data (1)
        return x, y
#%%8
# Early Stopping Class
class EarlyStopping:
    """주어진 patience 이후로 validation loss가 개선되지 않으면 학습을 조기 중지"""
    def __init__(self, patience=7, verbose=False, delta=0, path=args.cp_name):
        """
        Args:
            patience (int): validation loss가 개선된 후 기다리는 기간
                            Default: 7
            verbose (bool): True일 경우 각 validation loss의 개선 사항 메세지 출력
                            Default: False
            delta (float): 개선되었다고 인정되는 monitered quantity의 최소 변화
                            Default: 0
            path (str): checkpoint저장 경로
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_acc_max = 0
        self.val_f1_max = 0
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, valid_f1, valid_acc, train_f1, model, epoch):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, valid_f1, valid_acc, model)
        elif score <= self.best_score + self.delta:
            if epoch >= 10:
                self.counter += 1
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                print("Warming up... until 10 epoch")
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, valid_f1, valid_acc, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, valid_f1, valid_acc, model):
        '''validation loss가 감소하면 모델을 저장한다.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)

        self.val_loss_min = val_loss
        self.val_acc_max = valid_acc
        self.val_f1_max = valid_f1
#%%9
# sklearn.model_seleciton.ShuffleSplit를 이용한 train : validation set 분리
from sklearn.model_selection import ShuffleSplit

total_test_f1 = []
total_test_loss = []
total_test_acc = []
total_pred_his = []
total_yb_his = []

total_valid_f1 = []
total_valid_loss = []
total_valid_acc = []
total_valid_pred_his = []
total_valid_yb_his = []

#-----------------for test_idx in range(20)-----------------
# test subject 한 명 당 모델 성능 테스트
# test subject 설정
# test_idx = 0

for test_idx in range(20):
    
    
    # Leave-One-Out Cross Validation
    x1, y1  = s1_eeg, encoded_score1
    
    # test data    
    x1_test = x1[test_idx] # list (10)
    
    y1_test = y1[test_idx] # list (10)
    
    # train + validation data
    x1 = x1[0:test_idx] + x1[test_idx+1::] # list (19, 10)
    
    y1 = y1[0:test_idx] + y1[test_idx+1::] # list (19, 10)
    
    # train, validataion index 추출을 위한 배열 형성
    X = np.zeros(19)
    rs = ShuffleSplit(n_splits=1, random_state=test_idx+5, train_size=0.8, test_size=0.2)
    
    # relative index check
    for train_index, val_index in rs.split(X):
        print("TRAIN:", train_index, "Valid:", val_index)
    
    
    
    # EEG data (Session1 only)
    # train, validation split

    # train data: session1 데이터를 피험자 별로 연결, 총 150개 array 생성 (10*1*15)
    xdata, ydata = [], []
    for i in train_index:
        xdata = xdata + x1[i]   # s1_eeg (same subject) ([1, 2, ..., 10])
        ydata = ydata + y1[i]  # s1_score
    
    train_data = CustomDataset()
    train_dl = DataLoader(train_data, batch_size=2, shuffle=True)
    
    # validation
    xdata, ydata = [], []
    for j in val_index:
        xdata = xdata + x1[j]   # s1_eeg (same subject)
        ydata = ydata + y1[j]
        
    val_data = CustomDataset()
    val_dl = DataLoader(val_data, batch_size=1, shuffle=False)
    
    # test
    xdata, ydata = x1_test, y1_test
    test_data = CustomDataset()
    test_dl = DataLoader(test_data, batch_size=1, shuffle=False)
    
    
    # Load Model
    # change shallownet model by changing name ex) from braindecode.models.RSE_EEG_Model3
    from braindecode.models.RSE_EEG_Model2 import ShallowFBCSPNet
    
    from braindecode.util import set_random_seeds
    from torch.optim import AdamW, lr_scheduler
    import torch.nn as nn
    from sklearn.metrics import f1_score, accuracy_score
    
    # Accuracy metric: macro f1 score
    def score_function(real, pred):
        score = f1_score(real, pred)
        return score
    
    # Early Stopping, Epoch, LR scheduler setting
    patience = args.esp
    earlystopping = EarlyStopping(patience, verbose=True)
    epochs = args.n_epochs
    in_chans = args.n_ch_eeg
    input_time_length = args.input_time_length
    
    set_random_seeds(seed=42, cuda=device)
    model = ShallowFBCSPNet(in_chans=in_chans, n_classes=None, input_window_samples=None,
                            filter_time_length=27, pool_time_length=57,
                            pool_time_stride=11, final_conv_length=12,
                            pool_mode='mean', split_first_layer=True,
                            batch_norm=True, batch_norm_alpha=0.1, drop_prob=0.5)
    
    
    
    
    
    
    optimizer = AdamW(model.parameters(), lr=0.0625 * 0.01, weight_decay=0)
    scheduler = lr_scheduler.LambdaLR(optimizer=optimizer,
                                        lr_lambda=lambda epoch: 0.95 ** epoch, verbose=True)
    
    # loss function 설정
    criterion = nn.CrossEntropyLoss(weight=None, reduction='mean')
    scaler = torch.cuda.amp.GradScaler() 
    
    #-----------------------------Training-----------------------------------
    #------------------------------Start-------------------------------------
    model.to(device)
    train_loss_his = []
    val_loss_his = []
    
    
    # Training Start
    for epoch in range(epochs):
        start = time.time()
        train_loss = 0
        train_pred = []
        train_y = []
        
        valid_loss = 0
        valid_pred = []
        valid_y = []
        
        # train loop
        model.train()
        for idx, batch in enumerate(tqdm(train_dl)):
            xb = torch.tensor(batch[0], dtype=torch.float32, device=device).clone().detach()
            xb = xb.unsqueeze(dim=2)
    
            yb = torch.tensor(batch[1], dtype=torch.long, device=device).clone().detach()
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                pred = model(xb)
                loss = criterion(pred, yb)
                
                
            scaler.scale(loss).backward()
            
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()/len(train_dl)
            train_pred += pred.argmax(1).detach().cpu().numpy().tolist()
            train_y += yb.detach().cpu().numpy().tolist()
            
        train_loss_his.append(train_loss)
            
        # valid loop
        with torch.no_grad():
            model.eval()
            for batch in val_dl:
                xb = torch.tensor(batch[0], dtype=torch.float32, device=device).clone().detach()
                xb = xb.unsqueeze(dim=2)
                
                yb = torch.tensor(batch[1], dtype=torch.long, device=device).clone().detach()
                
                pred = model(xb)
                loss = criterion(pred, yb)
                
                
                valid_loss += loss.item()/len(val_dl)
                valid_pred += pred.argmax(1).detach().cpu().numpy().tolist()
                valid_y += yb.detach().cpu().numpy().tolist()
                
            val_loss_his.append(valid_loss)
            
        train_f1 = score_function(train_y, train_pred)
        train_acc = accuracy_score(train_y, train_pred)
        valid_f1 = score_function(valid_y, valid_pred)
        valid_acc = accuracy_score(valid_y, valid_pred)            
            
            
        TIME = time.time() - start
        print("============================================================================================")
        print(f'epoch : {epoch+1}/{epochs}    time : {TIME:.0f}s/{TIME*(epochs-epoch-1):.0f}s')
        print(f'TRAIN loss : {train_loss:.5f}, macro f1 : {train_f1:.5f}, accuracy : {train_acc:.5f}      VALID loss : {valid_loss:.5f}, macro f1 : {valid_f1:.5f}')
        print()
        
        earlystopping(valid_loss, valid_f1, valid_acc, train_f1, model, epoch)
        if earlystopping.early_stop:
            print("Early stopping")
            total_valid_acc.append(earlystopping.val_acc_max)
            total_valid_f1.append(earlystopping.val_f1_max)
            total_valid_loss.append(earlystopping.val_loss_min)
            break
        
        scheduler.step()
        
    # test loop
    model.load_state_dict(torch.load(args.cp_name))
    model.eval()
    test_loss = 0
    test_pred = []
    test_y = []
    pred_his = []
    yb_his = []
    
    with torch.no_grad():
        for batch in test_dl:
            xb = torch.tensor(batch[0], dtype=torch.float32, device=device).clone().detach()
            xb = xb.unsqueeze(dim=2)
            
            yb = torch.tensor(batch[1], dtype=torch.long, device=device).clone().detach()
            
            pred = model(xb)
            
            # pred check            
            loss = criterion(pred, yb)
            test_loss += loss.item()/len(test_dl)
            
            test_pred +=pred.argmax(1).detach().cpu().numpy().tolist()
            test_y += yb.detach().cpu().numpy().tolist()
            
    test_f1 = score_function(test_y, test_pred)
    test_acc = accuracy_score(test_y, test_pred)
            
    total_test_loss.append(test_loss)
    total_test_f1.append(test_f1)
    total_test_acc.append(test_acc)
    
    total_pred_his.append(test_pred)
    total_yb_his.append(test_y)
    
    print(f'{test_idx+1} subject TEST:    CE loss : {test_loss:.5f}      Macro F1 score : {test_f1:.5f}    Accuracy : {test_acc:.5f}') 
    print('Test Pred: ', test_pred, 'Test Target: ', test_y)   
    
#%%
mean_test_loss = np.mean(total_test_loss)
mean_test_f1 = np.mean(total_test_f1)
mean_test_acc = np.mean(total_test_acc)

mean_valid_acc = np.mean(total_valid_acc)
mean_valid_f1 = np.mean(total_valid_f1)
mean_valid_loss = np.mean(total_valid_loss)

#%%
# random model test
random_f1, random_acc = [], []
for l in total_yb_his:
    predict = np.random.randint(0, 9, size = 10)
    predict %= 2
    
    random_f1.append(score_function(l, predict))
    random_acc.append(accuracy_score(l, predict))

# calculate performance
mean_random_acc = np.mean(random_acc)
mean_random_f1 = np.mean(random_f1)

#%%
# Visualization of the last subject's loss
plt.plot(train_loss_his)
plt.plot(val_loss_his)
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Cross Entropy loss')
plt.legend(['Train', 'Valid'], loc='lower right')
plt.show()