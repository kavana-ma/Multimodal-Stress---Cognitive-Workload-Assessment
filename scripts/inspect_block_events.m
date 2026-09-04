%% INSPECT EEG EVENTS WITHIN N-BACK BLOCKS
%
% Purpose:
%   Inspect the stimulus/event structure inside the 27 N-back blocks.
%
% Subject:
%   VP001
%
% This script ONLY reads and prints information.
% It does NOT modify or save any data.

clear;
clc;

base = 'D:\major_project_group50\dataset';
subject = 'VP001';

%% Load EEG markers

file = fullfile(base,[subject '-EEG'],'mrk_nback.mat');
data = load(file);
mrk = data.mrk_nback;

%% Session marker codes

session_codes = [112 128 144];

session_labels = {'0-back','2-back','3-back'};

%% Extract all session markers

session_idx = [];

for k = 1:length(mrk.event.desc)

    if ismember(mrk.event.desc(k),session_codes)
        session_idx(end+1) = k;
    end

end

fprintf('\n');
fprintf('============================================================\n');
fprintf(' EEG N-BACK EVENT STRUCTURE: %s\n',subject);
fprintf('============================================================\n');

fprintf('\nTotal EEG markers: %d\n',length(mrk.event.desc));
fprintf('Session markers  : %d\n',length(session_idx));

%% Inspect every block

for b = 1:length(session_idx)

    idx_start = session_idx(b);

    if b < length(session_idx)
        idx_end = session_idx(b+1) - 1;
    else
        idx_end = length(mrk.event.desc);
    end

    session_code = mrk.event.desc(idx_start);

    class_idx = find(session_codes == session_code,1);
    class_name = session_labels{class_idx};

    session_time = mrk.time(idx_start)/1000;

    fprintf('\n------------------------------------------------------------\n');
    fprintf('BLOCK %02d | %s | start = %.3f sec\n', ...
        b,class_name,session_time);
    fprintf('------------------------------------------------------------\n');

    fprintf('%-6s %-12s %-12s %-12s\n', ...
        'Index','Code','Absolute(s)','Relative(s)');

    %% Print events inside block

    for k = idx_start:idx_end

        abs_time = mrk.time(k)/1000;
        rel_time = abs_time - session_time;

        fprintf('%-6d %-12d %-12.3f %-12.3f\n', ...
            k, ...
            mrk.event.desc(k), ...
            abs_time, ...
            rel_time);

    end

end

fprintf('\n============================================================\n');
fprintf('DONE\n');
fprintf('============================================================\n');