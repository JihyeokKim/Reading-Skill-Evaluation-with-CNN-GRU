%% EOG and EEG epoching
clc; close all; clear all; % 명령창, 작업공간 데이터, 띄운 Figure 등 초기화

%-------------------------------Data load---------------------------------%
load ./matfile/session1_eog.mat session1_eog
load ./matfile/session1_eeg.mat session1_eeg
load ./matfile/session2_eog.mat session2_eog
load ./matfile/session2_eeg.mat session2_eeg

load ./matfile/20_score1.mat score1
load ./matfile/20_score2.mat score2

%%
%--------------------------Data segmentation-------------------------------%
% constant setting
Fs_eog = 64;
Fs_eeg = 256;
subj_num = 20;          % 총 피험자 수
fp_script_num = 10;     % 각 session 당 지문 수

% Session1 - EOG
for i_subj = 1 : subj_num
    for i_script = 1 : fp_script_num
        tmp_data = session1_eog{i_subj, i_script};
%         tmp_data1 = zscore(tmp_data, 0, 2); % 전체 데이터에 대해 zscore
        
        i = 1;
        tmp_data_seq = [];
        tmp_data_seq1 = [];
        % while hop_length (1s) + window size (2s)<= eog data length,
        % braindecode shape에 맞춤
        while Fs_eog*(i-1) + Fs_eog*2 <= size(tmp_data, 2)
            tmp_data_seq(i,:,:) = zscore(tmp_data(:, 1+Fs_eog*(i-1) : Fs_eog*(i-1) + Fs_eog*2), 0, 2);
            i = i+1;
        end
        eog_s1_dataset{i_subj, i_script} = tmp_data_seq;
    end
end

% Session2 - EOG
for i_subj = 1 : subj_num
    for i_script = 1 : fp_script_num
        tmp_data = session2_eog{i_subj, i_script};
        i = 1;
        tmp_data_seq = [];
        
        % while hop_length (1s) + window size (2s) <= eog data length
        while Fs_eog*(i-1) + Fs_eog*2 <= size(tmp_data, 2)
            tmp_data_seq(i,:,:) = zscore(tmp_data(:, 1+Fs_eog*(i-1) : Fs_eog*(i-1) + Fs_eog*2), 0, 2);
            i = i+1;
        end
        eog_s2_dataset{i_subj, i_script} = tmp_data_seq;
    end
end
%%
% Session1 - EEG
for i_subj = 1 : subj_num
    for i_script = 1 : fp_script_num
        tmp_data = session1_eeg{i_subj, i_script};
        i = 1;
        tmp_data_seq = [];
        
        % while hop_length (1s) + window size (2s) <= eeg data length
        while Fs_eeg*(i-1) + Fs_eeg*2 <= size(tmp_data, 2)
            tmp_data_seq(i,:,:) = zscore(tmp_data(:, 1+Fs_eeg*(i-1) : Fs_eeg*(i-1) + Fs_eeg*2), 0, 2);
            i = i+1;
        end
        eeg_s1_dataset{i_subj, i_script} = tmp_data_seq;
    end
end

% Session2 - EEG
for i_subj = 1 : subj_num
    for i_script = 1 : fp_script_num
        tmp_data = session2_eeg{i_subj, i_script};
        i = 1;
        tmp_data_seq = [];
        
        % while hop_length (1s) + window size (2s) <= eeg data length
        while Fs_eeg*(i-1) + Fs_eeg*2 <= size(tmp_data, 2)
            tmp_data_seq(i,:,:) = zscore(tmp_data(:, 1+Fs_eeg*(i-1) : Fs_eeg*(i-1) + Fs_eeg*2), 0, 2);
            i = i+1;
        end
        eeg_s2_dataset{i_subj, i_script} = tmp_data_seq;
    end
end

%%
%---------------------------Label Preprocessing---------------------------%
% score, session의 지문순서(행) 랜덤 배열
% idx = randperm(10);
score.score1 = score1;
score.score1_norm = score1/7;
score.score1_err = score.score1_norm - mean(score.score1_norm, 2);
score.score1_std = std(score.score1_err, 0, 2);

score.score2 = score2;
score.score2_norm = score2/7;
score.score2_err = score.score2_norm - mean(score.score2_norm, 2);
score.score2_std = std(score.score2_err, 0, 2);

%%
%------------------------------Save Data----------------------------------%
cd('preprocessed_matfile')
save('preprocessed_eog_s1.mat', 'eog_s1_dataset', '-v7.3');
save('preprocessed_eeg_s1.mat', 'eeg_s1_dataset', '-v7.3');
save('preprocessed_eog_s2.mat', 'eog_s2_dataset', '-v7.3');
save('preprocessed_eeg_s2.mat', 'eeg_s2_dataset', '-v7.3');
save('preprocessed_score.mat', 'score', '-v7.3');
cd ..

% 이 파일들을 이용하여 model의 input으로 사용 (Without Augmentation model)