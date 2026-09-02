%% 01_load_data.m
% Member 3 - fNIRS Preprocessing
% TU Berlin Dataset

clc; clear;

% ---------------- DATA PATH ----------------
folder = 'C:\Users\Natar\OneDrive\Desktop\Major project\data\VP001-NIRS';

% Load N-back files
cnt = load(fullfile(folder,'cnt_nback.mat'));
mrk = load(fullfile(folder,'mrk_nback.mat'));
mnt = load(fullfile(folder,'mnt_nback.mat'));

% Extract structures
cntData = cnt.cnt_nback;
mrkData = mrk.mrk_nback;
mntData = mnt.mnt_nback;

% Extract HbO and HbR
HbO = cntData.oxy.x;
HbR = cntData.deoxy.x;

% Metadata
fs = cntData.oxy.fs;
channels = cntData.oxy.clab;

fprintf('\n===== DATA LOADED =====\n');
fprintf('Sampling Rate : %d Hz\n',fs);
fprintf('HbO Size      : %d x %d\n',size(HbO,1),size(HbO,2));
fprintf('HbR Size      : %d x %d\n',size(HbR,1),size(HbR,2));
fprintf('Channels       : %d\n',length(channels));

% Keep variables in workspace for next script
save('workspace_step1.mat');