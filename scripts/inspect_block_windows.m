%% INSPECT N-BACK BLOCK WINDOWS
%
% Purpose:
%   Inspect EEG and fNIRS N-back block boundaries before extraction.
%
% Subject:
%   VP001
%
% This script ONLY prints timing information.
% It does NOT modify or save data.

clear;
clc;

base = 'D:\major_project_group50\dataset';
subject = 'VP001';

%% Load markers

eeg_file = fullfile(base,[subject '-EEG'],'mrk_nback.mat');
nirs_file = fullfile(base,[subject '-NIRS'],'mrk_nback.mat');

eeg_data = load(eeg_file);
nirs_data = load(nirs_file);

eeg_mrk = eeg_data.mrk_nback;
nirs_mrk = nirs_data.mrk_nback;

%% EEG session markers

eeg_codes = [112 128 144];
eeg_labels = [0 2 3];

eeg_times = [];
eeg_class = [];

for k = 1:length(eeg_mrk.time)

    code = eeg_mrk.event.desc(k);

    idx = find(eeg_codes == code,1);

    if ~isempty(idx)

        eeg_times(end+1,1) = eeg_mrk.time(k)/1000;
        eeg_class(end+1,1) = eeg_labels(idx);

    end
end

%% NIRS session markers

nirs_codes = [7 8 9];
nirs_labels = [0 2 3];

nirs_times = [];
nirs_class = [];

for k = 1:length(nirs_mrk.time)

    code = nirs_mrk.event.desc(k);

    idx = find(nirs_codes == code,1);

    if ~isempty(idx)

        nirs_times(end+1,1) = nirs_mrk.time(k)/1000;
        nirs_class(end+1,1) = nirs_labels(idx);

    end
end

%% Load signal lengths

eeg_cnt_file = fullfile(base,[subject '-EEG'],'cnt_nback.mat');
nirs_cnt_file = fullfile(base,[subject '-NIRS'],'cnt_nback.mat');

eeg_cnt_data = load(eeg_cnt_file);
nirs_cnt_data = load(nirs_cnt_file);

cnt_eeg = eeg_cnt_data.cnt_nback;
cnt_nirs = nirs_cnt_data.cnt_nback;

eeg_fs = cnt_eeg.fs;
nirs_fs = cnt_nirs.oxy.fs;

eeg_duration = size(cnt_eeg.x,1) / eeg_fs;
nirs_duration = size(cnt_nirs.oxy.x,1) / nirs_fs;

%% Print

fprintf('\n============================================================\n');
fprintf('N-BACK BLOCK WINDOW INSPECTION: %s\n',subject);
fprintf('============================================================\n');

fprintf('\nEEG sampling rate  : %.2f Hz\n',eeg_fs);
fprintf('fNIRS sampling rate: %.2f Hz\n',nirs_fs);

fprintf('EEG duration       : %.3f sec\n',eeg_duration);
fprintf('fNIRS duration     : %.3f sec\n',nirs_duration);

fprintf('\n');
fprintf('%-6s %-8s %-12s %-12s %-12s %-12s\n', ...
    'Block','Class','EEG start','NIRS start','Offset','EEG gap');

fprintf('------------------------------------------------------------\n');

for b = 1:27

    eeg_start = eeg_times(b);
    nirs_start = nirs_times(b);

    offset = nirs_start - eeg_start;

    if b < 27
        eeg_gap = eeg_times(b+1) - eeg_times(b);
    else
        eeg_gap = NaN;
    end

    fprintf('%-6d %-8d %-12.3f %-12.3f %-12.3f %-12.3f\n', ...
        b, ...
        eeg_class(b), ...
        eeg_start, ...
        nirs_start, ...
        offset, ...
        eeg_gap);

end

fprintf('\n============================================================\n');
fprintf('RUN BOUNDARIES\n');
fprintf('============================================================\n');

fprintf('\nRun 1: blocks 1-9\n');
fprintf('  EEG : %.3f -> %.3f sec\n', ...
    eeg_times(1),eeg_times(10));

fprintf('  NIRS: %.3f -> %.3f sec\n', ...
    nirs_times(1),nirs_times(10));

fprintf('\nRun 2: blocks 10-18\n');
fprintf('  EEG : %.3f -> %.3f sec\n', ...
    eeg_times(10),eeg_times(19));

fprintf('  NIRS: %.3f -> %.3f sec\n', ...
    nirs_times(10),nirs_times(19));

fprintf('\nRun 3: blocks 19-27\n');
fprintf('  EEG : %.3f -> %.3f sec\n', ...
    eeg_times(19),eeg_times(27));

fprintf('  NIRS: %.3f -> %.3f sec\n', ...
    nirs_times(19),nirs_times(27));

fprintf('\nDone.\n');