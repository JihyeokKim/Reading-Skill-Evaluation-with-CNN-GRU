%% 초기화 / 데이터 로드
% Augmentation data를 이용하려는 경우 01 preprocessing 실행 후 이 파일 실행

clear; close all; clc;
addpath('functions')
addpath(genpath('CalibrationTimeReductionProcIEEE2015'))

% data load (S1 EEG data augmentation)
load('matfile/session1_eeg.mat');     % 32 channel eeg data
load('encoded_score1b.mat');

[n_subject, n_paragraph] = size(session1_eeg);

% windowLength: the length of the STFT windows. the fourier transformed windows will be mixed between
%   trials to generate artificial new trials.
% stepSize: the number of samples between two consecutive STFT windows. (if stepSize < windowLength the windows will overlap)
% nbData: the number of artifical EEG trials desired, for each class
% Fs: data sampling frequency
Fs = 256;
windowLength = 128;
stepSize = windowLength/2;
nbData = 10; % 각 epoch 마다를 기준으로 10 배수

% Input:
% EEGSignals: EEG signals, composed of 2 classes. These signals
%   are a structure such that:
%       EEGSignals.x: the EEG signals as a [Ns * Nc * Nt] Matrix where
%           Ns: number of EEG samples per trial
%           Nc: number of channels (EEG electrodes)
%           nT: number of trials
%       EEGSignals.y: a [1 * Nt] vector containing the class labels for each trial
%       EEGSignals.s: the sampling frequency (in Hz)

%% 데이터 preprocessing
% 한 피험자 데이터를 3등분 (trials 수 = 3)해서 나머지는 날림
for i_subject = 1 : n_subject
    for i_session = 1 : n_paragraph
        x = []; y = [];
        tmp = session1_eeg{i_subject, i_session};
        [c, l] = size(tmp);
        l = l - mod(l,3);
        tmp = tmp.';
        
        
        for i = 1 : 3
            x(:,:,i) = tmp(1+l*(i-1)/3:l*i/3, :);
            y(:,i) = score1(i_subject, i_session);
        end
        
        % x = [samples 수 * channel 수 * trials 수 (3)]
        % y = [score label, score label, score label] (모두 같음)
        % s = sampling frequency (256)
        EEGSignals.x = x;
        EEGSignals.y = y;
        EEGSignals.s = Fs;
        
        artificialDataSet = generateArtificialDataSTFT_rev2(EEGSignals, windowLength, stepSize, nbData); % stft dataset 생성
        augmented_eeg_data{i_subject, i_session} = artificialDataSet.x;
        augmented_score_class{i_subject, i_session} = artificialDataSet.y;
    end
end

%% 데이터 저장
cd("augmented_matfile")
% augmented session 1 eeg data with 32 channel for binary classification
save('augmented_eeg32_bs1.mat', 'augmented_eeg_data', '-v7.3');
save('augmented_eeg_bs1_score.mat', 'augmented_score_class', '-v7.3');
cd ..

% 03_epoching으로 go