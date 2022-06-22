%% Initial setting
clc; close all; clear all;
addpath(genpath('./03_braindecode_matlab/eeglab11_0_5_4b/'));
addpath(genpath('./03_braindecode_matlab/Online/'));
addpath(genpath('./03_braindecode_matlab/functions/'));


% DB의 파일, 폴더 나열
path = './2021_KJH_Reading_evaluation/';
list_subj = dir('./2021_KJH_Reading_evaluation/');
list_subj = list_subj(3:22);

%% Parameters                   
par.n_subj = length(list_subj);                                    % 20명
par.fp_script_num = 10;                                                 % 각 session 당 지문 수

par.freq_Bio = 2048;                                                    % 기기 Sampling rate (Biosemi는 2048 Hz)
par.freq_eog = 64;
par.freq_eeg = 256;
par.freq_HPF = 0.1;
par.freq_LPF = 55;

[para_LPF_b, para_LPF_a] = butter(4, 2*par.freq_LPF/par.freq_Bio, 'low');
[para_HPF_b, para_HPF_a] = butter(4, 2*par.freq_HPF/par.freq_Bio, 'high');
para_DS_eog = online_downsample_init(par.freq_Bio/par.freq_eog);                % Downsample 해주는 Manual 함수
para_DS_eeg = online_downsample_init(par.freq_Bio/par.freq_eeg);

% 저장공간
session1_eog = cell(par.n_subj, par.fp_script_num);
session2_eog = cell(par.n_subj, par.fp_script_num);
session1_eeg = cell(par.n_subj, par.fp_script_num);
session2_eeg = cell(par.n_subj, par.fp_script_num);
%% Pre-processing for subject 11 ~ 23
% subject 11 ~ 23 은 test.01.bdf 파일에 session1, session2 데이터 모두 존재
% session 1의 경우 Script+Problem trigger (11~20) 의 2번째 ~ 3번째 사이 데이터가 생체 신호 데이터
% session 2의 경우 Script/Problem trigger (21~30) 의
% subject 11, 12 의 경우: 2번째 ~ 4번째 사이 데이터가 생체 신호 데이터
% subject 13 ~ 23 의 경우: 2번째 ~ 3번째 사이 데이터가 생체 신호 데이터 이용

for i_subj = 1: 13
    % bdf load
    temp_eog_load = pop_biosig(fullfile([path,list_subj(i_subj).name,'/','test01.bdf']));    % bdf파일 로드
    temp_trig = temp_eog_load.urevent;
    temp_trig_cell = struct2cell(temp_trig);
    temp_trig_mat = cell2mat(temp_trig_cell);

    % Matrix화 된 trigger data
    trig_raw = permute(temp_trig_mat, [1 3 2]); %2x1x106 -> 2x106(x1)
    
    % Trigger 정렬
    % 1~18 Cali1 (18)
    
    % 19~48 Session1 (10 scripts)	// 11-20 : 10개 *3
    temp_session1 = sortrows(trig_raw(:,19:18+3*par.fp_script_num)');   % 첫 번째 열의 요소를 기준으로 행렬의 행을 오름차순으로 정렬/ if 반복) 다음 열에 따라 정렬
    temp2_session1 = reshape(temp_session1(:,2), 3, par.fp_script_num)';
    trigger_session1_ds_eog = round(temp2_session1 * par.freq_eog / par.freq_Bio);  % round : 반올림
    trigger_session1_ds_eeg = round(temp2_session1 * par.freq_eeg / par.freq_Bio);
    
    % Session2 (Script / Problem 에서 각 지문별 trigger 위치 저장)
    % (지문 번호 오름차순으로 정렬)
    temp_session2 = sortrows(trig_raw(:, 67: 66 + 4*par.fp_script_num)');
    temp2_session2 = reshape(temp_session2(:, 2), 4, par.fp_script_num)';
    trigger_session2_ds_eog = round(temp2_session2 * par.freq_eog / par.freq_Bio);
    trigger_session2_ds_eeg = round(temp2_session2 * par.freq_eeg / par.freq_Bio);
    
    % Raw EOG Data
    eog_raw = double(temp_eog_load.data(33:37,:)');             % EOG 채널 뽑기
    eog_LPF = filtfilt(para_LPF_b, para_LPF_a, eog_raw);        % LPF 통과
    eog_HPF = filtfilt(para_HPF_b, para_HPF_a, eog_LPF);        % HPF 통과
    [~, eog_DS] = online_downsample_apply(para_DS_eog, eog_HPF');   % 2048 Hz > 64 Hz Downsampling
    
    % Raw EEG Data
    eeg_raw = double(temp_eog_load.data(1:32,:)');             % 채널 뽑아내기
    eeg_LPF = filtfilt(para_LPF_b, para_LPF_a, eeg_raw);        % LPF 통과
    eeg_HPF = filtfilt(para_HPF_b, para_HPF_a, eeg_LPF);        % HPF 통과
    [~, eeg_DS] = online_downsample_apply(para_DS_eeg, eeg_HPF');   % 2048 Hz > 256 Hz Downsampling

     %--------------------------Segmentation---------------------------------%        
    for i_script = 1 : par.fp_script_num  % 1부터 10까지
        % Session 1 - trigger 2th ~ 3th
        t_start_1_eog = trigger_session1_ds_eog(i_script,2);
        t_end_1_eog = trigger_session1_ds_eog(i_script,3);
        t_start_1_eeg = trigger_session1_ds_eeg(i_script,2);
        t_end_1_eeg = trigger_session1_ds_eeg(i_script,3);
        
        session1_eog{i_subj, i_script} = eog_DS(t_start_1_eog : t_end_1_eog, :)';
        session1_eeg{i_subj, i_script} = eeg_DS(t_start_1_eeg : t_end_1_eeg, :)';
        
        % Session 2 - trigger 2th ~ 4th for subject 11, 12
        %                     2th ~ 3th for subject 13 ~ 23
        if i_subj <= 2
            t_start_2_eog = trigger_session2_ds_eog(i_script,2);
            t_end_2_eog = trigger_session2_ds_eog(i_script, 4);
            t_start_2_eeg = trigger_session2_ds_eeg(i_script,2);
            t_end_2_eeg = trigger_session2_ds_eeg(i_script, 4);
        else
            t_start_2_eog = trigger_session2_ds_eog(i_script,2);
            t_end_2_eog = trigger_session2_ds_eog(i_script, 3);
            t_start_2_eeg = trigger_session2_ds_eeg(i_script,2);
            t_end_2_eeg = trigger_session2_ds_eeg(i_script, 3);
        end
        
        session2_eog{i_subj, i_script} = eog_DS(t_start_2_eog : t_end_2_eog, :)';
        session2_eeg{i_subj, i_script} = eeg_DS(t_start_2_eeg : t_end_2_eeg, :)';
    end
    
    % 진행상황 출력
    text = (i_subj) + "번째 피험자";
    disp(text)
end
%% Pre-processing for subject 24 ~ 30
% subject 24 ~ 30 은 test01.bdf 파일에 session1, test02.bdf 파일에 session2 데이터 존재
% session 1의 경우 Script+Problem trigger (11~20) 의 2번째 ~ 3번째 사이 데이터가 생체 신호 데이터
% session 2의 경우 Script/Problem trigger (21~30) 의 2번째 ~ 3번째 사이 데이터가 생체 신호 데이터

for i_subj = 14 : 20

    % bdf load
    temp_eog_load = pop_biosig(fullfile([path,list_subj(i_subj).name,'/','test01.bdf']));    % bdf파일 로드
    temp_trig = temp_eog_load.urevent;
    temp_trig_cell = struct2cell(temp_trig);
    temp_trig_mat = cell2mat(temp_trig_cell);
    
    temp_eog_load2 = pop_biosig(fullfile([path,list_subj(i_subj).name,'/','test02.bdf']));
    temp_trig2 = temp_eog_load2.urevent;
    temp_trig_cell2 = struct2cell(temp_trig2);
    temp_trig_mat2 = cell2mat(temp_trig_cell2);
    
    % Matrix화 된 trigger data
    trig_raw = permute(temp_trig_mat, [1 3 2]); %2x1x106 -> 2x106(x1)
    trig_raw2 = permute(temp_trig_mat2, [1 3 2]);
    
    % Trigger 정렬
    % 1~18 Cali1 (18)
    
    % 19~48 Session1 (10 scripts)	// 11-20 : 10개 *3
    temp_session1 = sortrows(trig_raw(:,19:18+3*par.fp_script_num)');   % 첫 번째 열의 요소를 기준으로 행렬의 행을 오름차순으로 정렬/ if 반복) 다음 열에 따라 정렬
    temp2_session1 = reshape(temp_session1(:,2),3,par.fp_script_num)';
    trigger_session1_ds_eog = round(temp2_session1 * par.freq_eog / par.freq_Bio);  % round : 반올림
    trigger_session1_ds_eeg = round(temp2_session1 * par.freq_eeg / par.freq_Bio);
        
    % Raw EOG Data
    eog_raw = double(temp_eog_load.data(33:37,:)');             % EOG 채널 뽑기
    eog_LPF = filtfilt(para_LPF_b, para_LPF_a, eog_raw);        % LPF 통과
    eog_HPF = filtfilt(para_HPF_b, para_HPF_a, eog_LPF);        % HPF 통과
    [~, eog_DS] = online_downsample_apply(para_DS_eog, eog_HPF');   % 2048 Hz > 64 Hz Downsampling
    
    % Raw EEG Data
    eeg_raw = double(temp_eog_load.data(1:32,:)');             % 채널 뽑아내기
    eeg_LPF = filtfilt(para_LPF_b, para_LPF_a, eeg_raw);        % LPF 통과
    eeg_HPF = filtfilt(para_HPF_b, para_HPF_a, eeg_LPF);        % HPF 통과
    [~, eeg_DS] = online_downsample_apply(para_DS_eeg, eeg_HPF');   % 2048 Hz > 256 Hz Downsampling
    
    % Session2 (Script / Problem 에서 각 지문별 trigger 위치 저장)
    % (지문 번호 오름차순으로 정렬)
    temp_session2 = sortrows(trig_raw2(:, 19: 18 + 4*par.fp_script_num)');
    temp2_session2 = reshape(temp_session2(:, 2), 4, par.fp_script_num)';
    trigger_session2_ds_eog = round(temp2_session2 * par.freq_eog / par.freq_Bio);
    trigger_session2_ds_eeg = round(temp2_session2 * par.freq_eeg / par.freq_Bio);
    
    % Raw EOG Data
    eog_raw2 = double(temp_eog_load2.data(33:37,:)');             % EOG 채널 뽑기
    eog_LPF2 = filtfilt(para_LPF_b, para_LPF_a, eog_raw2);        % LPF 통과
    eog_HPF2 = filtfilt(para_HPF_b, para_HPF_a, eog_LPF2);        % HPF 통과
    [~, eog_DS2] = online_downsample_apply(para_DS_eog, eog_HPF2');   % 2048 Hz > 64 Hz Downsampling
    
    % Raw EEG Data
    eeg_raw2 = double(temp_eog_load2.data(1:32,:)');             % 채널 뽑아내기
    eeg_LPF2 = filtfilt(para_LPF_b, para_LPF_a, eeg_raw2);        % LPF 통과
    eeg_HPF2 = filtfilt(para_HPF_b, para_HPF_a, eeg_LPF2);        % HPF 통과
    [~, eeg_DS2] = online_downsample_apply(para_DS_eeg, eeg_HPF2');   % 2048 Hz > 256 Hz Downsampling
    %--------------------------Segmentation---------------------------------%
    for i_script = 1 : par.fp_script_num  % 1부터 10까지
        % Session 1 - trigger 2th ~ 3th
        t_start_1_eog = trigger_session1_ds_eog(i_script,2);
        t_end_1_eog = trigger_session1_ds_eog(i_script,3);
        t_start_1_eeg = trigger_session1_ds_eeg(i_script,2);
        t_end_1_eeg = trigger_session1_ds_eeg(i_script,3);
        
        session1_eog{i_subj, i_script} = eog_DS(t_start_1_eog : t_end_1_eog, :)';
        session1_eeg{i_subj, i_script} = eeg_DS(t_start_1_eeg : t_end_1_eeg, :)';
        
        % Session 2 - trigger 2th ~ 3th for subject 24 ~ 30
        
        t_start_2_eog = trigger_session2_ds_eog(i_script,2);
        t_end_2_eog = trigger_session2_ds_eog(i_script, 3);
        t_start_2_eeg = trigger_session2_ds_eeg(i_script,2);
        t_end_2_eeg = trigger_session2_ds_eeg(i_script, 3);
        
        session2_eog{i_subj, i_script} = eog_DS2(t_start_2_eog : t_end_2_eog, :)';
        session2_eeg{i_subj, i_script} = eeg_DS2(t_start_2_eeg : t_end_2_eeg, :)';
    end
    
    text = (i_subj) + "번째 피험자";
    disp(text)
end

%%
cd("matfile")
save('session1_eog.mat', 'session1_eog', '-v7.3');
save('session1_eeg.mat', 'session1_eeg', '-v7.3');
save('session2_eog.mat', 'session2_eog', '-v7.3');
save('session2_eeg.mat', 'session2_eeg', '-v7.3');
cd ..
