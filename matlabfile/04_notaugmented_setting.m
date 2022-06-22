%% 데이터 preprocessing
% augmented data를 이용한 model test에서 validation & test data로 쓰일 데이터를 생성하는 코드
% python main file을 통해 encoded_score1b.mat 파일 (threshold를 기준으로 0 = bad, 1 = good으로 분류된 score 파일)을 우선적으로 만들어야 함 (해당 코드는 삭제됨)

clc; close all; clear all;
load('matfile/session1_eeg.mat');     % 32 channel eeg data
load('encoded_score1b.mat');    % binary classification score data (score1 변수 생성)

[n_subject, n_paragraph] = size(score1);

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
        
        
        not_augmented_eeg_data{i_subject, i_session} = x;
        not_augmented_score_class{i_subject, i_session} = y;
    end
end

%%
%--------------------------Data segmentation-------------------------------%
% constant setting
Fs_eog = 64;
Fs_eeg = 256;
subj_num = 20;          % 총 피험자 수
fp_script_num = 10;     % 각 session 당 지문 수
n_at = 3;              % augmented data trial 수

%--------------------------Data permutation--------------------------------
for i_subj = 1 : subj_num
    for i_script = 1 : fp_script_num
        not_augmented_score_class{i_subj, i_script} = not_augmented_score_class{i_subj, i_script}.';
    end
end

% Session1 - EEG
for i_subj = 1 : subj_num
    for i_script = 1 : fp_script_num
        tmp_data = not_augmented_eeg_data{i_subj, i_script};
        tmp_data_seq_ar = [];
        for i_trial = 1 : n_at
            tmp_data2 = tmp_data(:, :, i_trial).';
            
            i = 1;
            tmp_data_seq = [];
            
            % while hop_length (1s) + window size (2s) <= eeg data length
            while Fs_eeg*(i-1) + Fs_eeg*2 <= size(tmp_data2, 2)
                tmp_data_seq(i,:,:) = zscore(tmp_data2(:, 1+Fs_eeg*(i-1) : Fs_eeg*(i-1) + Fs_eeg*2), 0, 2);
                i = i+1;
            end
            tmp_data_seq_ar(i_trial,:,:,:) = tmp_data_seq;
        end
        
        eeg_s1_dataset{i_subj, i_script} = tmp_data_seq_ar;

    end
end

%%
% validation, test data로 쓰일 data 저장
cd('preprocessed_matfile_aug')
save('not_augmented_eeg_32s1.mat', 'eeg_s1_dataset', '-v7.3');
save('not_augmented_eeg_bs1_score.mat', 'not_augmented_score_class', '-v7.3');
cd ..