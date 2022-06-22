%% EOG and EEG epoching
clc; close all; clear all; % 명령창, 작업공간 데이터, 띄운 Figure 등 초기화

% Augmented S1 EEG data에 대한 code
%-------------------------------Data load---------------------------------%
load ./augmented_matfile/augmented_eeg32_bs1.mat
load ./augmented_matfile/augmented_eeg_bs1_score.mat

%%
%--------------------------Data segmentation-------------------------------%
% constant setting
Fs_eog = 64;
Fs_eeg = 256;
subj_num = 20;          % 총 피험자 수
fp_script_num = 10;     % 각 session 당 지문 수
n_at = 30;              % augmented data trial 수 ((trial=)3 * (nbData=)10)

%--------------------------Data permutation--------------------------------
for i_subj = 1 : subj_num
    for i_script = 1 : fp_script_num
        augmented_score_class{i_subj, i_script} = augmented_score_class{i_subj, i_script}.';
    end
end

%%
% Session1 - EEG
for i_subj = 1 : subj_num
    for i_script = 1 : fp_script_num
        tmp_data = augmented_eeg_data{i_subj, i_script};
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
%------------------------------Save Data----------------------------------%
cd('preprocessed_matfile_aug')
save('preprocessed_augmented_eeg_32bs1.mat', 'eeg_s1_dataset', '-v7.3');
save('preprocessed_augmented_bscore.mat', 'augmented_score_class', '-v7.3');
cd ..

% 04_notaugmented_setting을 통해 validation, test로 쓸 데이터 따로 생성
