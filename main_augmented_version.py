"""
EEG-based Reading Skill Evaluation Model Design with Deep Learning Architecture
- Binary Classification based on the threshold
- Input: Zero-padded and augmented EEG data
- Target: Encoded and augmented score error data (0 if <thereshold, 1 if >=threshold)
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
                    default="RSE_EEG_S1_BinaryClassification_Model_With_Padding_Augmentation", help="name of model")
parser.add_argument("--mat_dir", type=str,
                    default="./preprocessed_matfile_aug/", help="directory of matfiles")
parser.add_argument("--n_ch_eeg", type=int,
                    default=32, help="the number of EEG channels")
parser.add_argument("--n_ch_eog", type=int,
                    default=5, help="the number of EOG channels")
parser.add_argument("--n_filters", type=int,
                    default=40, help="the number of filters")
parser.add_argument("--input_time_length", type=int,
                    default=512, help="the window size")
parser.add_argument("--esp", type=int,
                    default=5, help="the patience of early stopping")
parser.add_argument("--n_epochs", type=int,
                    default=100, help="the maximum number of epochs")
parser.add_argument("--trainwithRMSE", type=bool,
                    default=False, help="setting the train loss as RMSE or MSE")
parser.add_argument("--cp_name", type=str,
                    default='checkpointEEGS1_binary_classification_aug.pt', help="the filename of checkpoint")
parser.add_argument("--makeclassweight", type=bool,
                    default=False, help="setting the class weight")
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
# matfile 불러오기
file_list = os.listdir(args.mat_dir)
load_list = []
for file in file_list:
    filename = args.mat_dir + file
    data = mat73.loadmat(filename)
    load_list.append(data)
    
#%%4
# x data
s1_eeg = load_list[3]['eeg_s1_dataset']
s1_test_eeg = load_list[0]['eeg_s1_dataset']

# label data
s1_score_class = load_list[2]['augmented_score_class']
s1_test_score_class = load_list[1]['not_augmented_score_class']

#%% - Padding (S1 EEG)
n_ch = args.n_ch_eeg
n_wl = 512

max_length = 0
for i in s1_eeg:
    for j in i:
        if j.shape[1] >= max_length:
            max_length = j.shape[1]
            
for idx1, i in enumerate(s1_eeg):
    for idx2, j in enumerate(i):
        if j.shape[1] < max_length:
            n = max_length - j.shape[1]
            j = np.concatenate((np.zeros((30, n, n_ch, n_wl), dtype=np.float32), j), axis=1)
        s1_eeg[idx1][idx2] = j
            
for idx1, i in enumerate(s1_test_eeg):
    for idx2, j in enumerate(i):
        if j.shape[1] < max_length:
            n = max_length - j.shape[1]
            j = np.concatenate((np.zeros((3, n, n_ch, n_wl), dtype=np.float32), j), axis=1)
        s1_test_eeg[idx1][idx2] = j
        
#%% - append
augmented_s1_eeg = [[] for x in range(20)]
not_augmented_s1_eeg = [[] for x in range(20)]

for idx1, i in enumerate(s1_eeg):
    for idx2, j in enumerate(i):
        for k in range(30):
            augmented_s1_eeg[idx1].append(j[k])
del s1_eeg

for idx1, i in enumerate(s1_test_eeg):
    for idx2, j in enumerate(i):
        for k in range(3):
            not_augmented_s1_eeg[idx1].append(j[k])

del s1_test_eeg

#%%
augmented_s1_score_class = [[] for x in range(20)]
for idx1, i in enumerate(s1_score_class):
    for idx2, j in enumerate(i):
        for k in j:
            augmented_s1_score_class[idx1].append(k)

not_augmented_s1_score_class = [[] for x in range(20)]
for idx1, i in enumerate(s1_test_score_class):
    for idx2, j in enumerate(i):
        for k in j:
            not_augmented_s1_score_class[idx1].append(k)
            
#%%7
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.x_data = xdata    
        self.y_data = ydata
        
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        x = self.x_data[idx]     # 리턴 x 데이터 (sequence 길이, 32, 512)
        y = self.y_data[idx]    # 리턴 y 데이터 (1)
       
        # y = np.asarray(y, dtype='float32')
        return x, y
#%%8
# Early Stopping 클래스 선언
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
        self.delta = delta
        self.path = path
        self.val_f1_max = 0
        self.val_acc_max = 0

    def __call__(self, val_loss, valid_f1, train_f1, valid_acc, model, epoch):

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
        self.val_f1_max = valid_f1
        self.val_acc_max = valid_acc
#%%8
# sklearn.model_seleciton.ShuffleSplit를 이용한 train : validation set 분리
from sklearn.model_selection import ShuffleSplit

total_test_f1 = []
total_test_loss = []
total_test_acc = []
total_pred_his = []
total_yb_his = []

total_valid_f1 = []
total_valid_acc = []
total_valid_loss = []

#-------------------for test_idx in range(20)------------------
# test subject 한 명 당 모델 성능 테스트
# test subject 설정
# test_idx = 0

for test_idx in range(20):
    
    
    # leave-one-out data 생성 
    x1, y1 = augmented_s1_eeg, augmented_s1_score_class
    x1_notaug, y1_notaug = not_augmented_s1_eeg, not_augmented_s1_score_class
    
    # test data    
    x1_test = x1_notaug[test_idx] # list (10)

    
    y1_test = y1_notaug[test_idx] # list (10)

    
    # train + validation data
    x1 = x1[0:test_idx] + x1[test_idx+1::] # list (19, 10)

    
    y1 = y1[0:test_idx] + y1[test_idx+1::] # list (19, 10)

    
    # train, validataion index 추출을 위한 배열 형성
    X = np.zeros(19)
    rs = ShuffleSplit(n_splits=1, random_state=test_idx+5, train_size=0.8, test_size=0.2)
    
    # index check (절대 index는 아님)
    for train_index, val_index in rs.split(X):
        print("TRAIN:", train_index, "Valid:", val_index)
    
    
    
    # Model 1-2 EEG data (Session1 only)
    # train, validation split
    # xdata, ydata 생성
    # xdata: train subjects의 [s1_eeg data]
    # ydata: train subjects의 [s1_score_err]
    
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
        xdata = xdata + x1_notaug[j]   # s1_eeg (same subject)
        ydata = ydata + y1_notaug[j]
        
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
    
    # Early Stopping, Epoch, LR scheduler 설정
    patience = args.esp
    earlystopping = EarlyStopping(patience, verbose=True)
    epochs = args.n_epochs
    in_chans = args.n_ch_eeg
    input_time_length = args.input_time_length
    
    set_random_seeds(seed=42, cuda=device)
    model = ShallowFBCSPNet(in_chans=in_chans, n_classes=None, input_window_samples=None, n_filters_time=40,
                            filter_time_length=27, n_filters_spat=40, pool_time_length=57,
                            pool_time_stride=11, final_conv_length=12,
                            pool_mode='mean', split_first_layer=True,
                            batch_norm=True, batch_norm_alpha=0.1, drop_prob=0.5)
    
    
    
    
    
    

    optimizer = AdamW(model.parameters(), lr=0.0625 * 0.01, weight_decay=0)
    scheduler = lr_scheduler.LambdaLR(optimizer=optimizer,
                                        lr_lambda=lambda epoch: 0.95 ** epoch, verbose=True)
    # loss function 설정
    criterion = nn.CrossEntropyLoss(weight=None, reduction='sum')
    
   
    
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
            
            # pred 값 확인            
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


mean_valid_loss = np.mean(total_valid_loss)
mean_valid_f1 = np.mean(total_valid_f1)
mean_valid_acc = np.mean(total_valid_acc)

#%%
# Visualization of the last subject's loss
plt.plot(train_loss_his)
plt.plot(val_loss_his)
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Cross Entropy loss')
plt.legend(['Train', 'Valid'], loc='lower right')
plt.show()