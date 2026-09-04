%% VERIFY EEG-fNIRS N-BACK BLOCK ALIGNMENT
%
% Purpose:
%   Verify the temporal relationship between EEG and fNIRS
%   N-back session markers for all paired subjects.
%
% Does NOT extract epochs.
% Does NOT modify data.
%
% Classes:
%   EEG 112 = 0-back
%   EEG 128 = 2-back
%   EEG 144 = 3-back
%
%   NIRS 7 = 0-back
%   NIRS 8 = 2-back
%   NIRS 9 = 3-back

clear;
clc;

%% Paths

base = 'D:\major_project_group50\dataset';

paired_subjects = { ...
    'VP001','VP002','VP003','VP004','VP005','VP006', ...
    'VP007','VP008','VP009','VP010','VP011', ...
    'VP014','VP015','VP016','VP017','VP018','VP019', ...
    'VP020','VP021','VP022','VP023','VP024','VP025','VP026'};

%% Marker mappings

eeg_codes  = [112 128 144];
nirs_codes = [7   8   9];

class_labels = [0 2 3];

%% Results

results = [];

fprintf('\n============================================\n');
fprintf(' EEG-fNIRS N-BACK BLOCK ALIGNMENT CHECK\n');
fprintf('============================================\n\n');

for s = 1:length(paired_subjects)

    subject = paired_subjects{s};

    fprintf('\n--------------------------------------------\n');
    fprintf('%s\n', subject);
    fprintf('--------------------------------------------\n');

    eeg_folder  = fullfile(base, [subject '-EEG']);
    nirs_folder = fullfile(base, [subject '-NIRS']);

    %% Check files

    eeg_file  = fullfile(eeg_folder, 'mrk_nback.mat');
    nirs_file = fullfile(nirs_folder, 'mrk_nback.mat');

    if ~isfile(eeg_file)
        fprintf('ERROR: EEG marker file missing\n');
        continue;
    end

    if ~isfile(nirs_file)
        fprintf('ERROR: NIRS marker file missing\n');
        continue;
    end

    %% Load

    eeg_data  = load(eeg_file);
    nirs_data = load(nirs_file);

    eeg_mrk  = eeg_data.mrk_nback;
    nirs_mrk = nirs_data.mrk_nback;

    %% EEG session markers

    eeg_times = [];
    eeg_class = [];

    for k = 1:length(eeg_mrk.time)

        code = eeg_mrk.event.desc(k);

        idx = find(eeg_codes == code, 1);

        if ~isempty(idx)

            eeg_times(end+1,1) = eeg_mrk.time(k) / 1000;
            eeg_class(end+1,1) = class_labels(idx);

        end
    end

    %% NIRS session markers

    nirs_times = [];
    nirs_class = [];

    for k = 1:length(nirs_mrk.time)

        code = nirs_mrk.event.desc(k);

        idx = find(nirs_codes == code, 1);

        if ~isempty(idx)

            nirs_times(end+1,1) = nirs_mrk.time(k) / 1000;
            nirs_class(end+1,1) = class_labels(idx);

        end
    end

    %% Print counts

    fprintf('EEG blocks  : %d\n', length(eeg_times));
    fprintf('NIRS blocks : %d\n', length(nirs_times));

    %% Match by class + temporal order

    n = min(length(eeg_times), length(nirs_times));

    if n == 0
        fprintf('WARNING: No blocks found.\n');
        continue;
    end

    eeg_use  = eeg_times(1:n);
    nirs_use = nirs_times(1:n);

    eeg_cls  = eeg_class(1:n);
    nirs_cls = nirs_class(1:n);

    %% Check class sequence

    class_match = eeg_cls == nirs_cls;

    fprintf('Class sequence matches: %d / %d\n', ...
        sum(class_match), n);

    %% Calculate offset

    offsets = nirs_use - eeg_use;

    fprintf('Median NIRS-EEG offset: %.3f sec\n', ...
        median(offsets));

    fprintf('Mean NIRS-EEG offset  : %.3f sec\n', ...
        mean(offsets));

    fprintf('Min offset             : %.3f sec\n', ...
        min(offsets));

    fprintf('Max offset             : %.3f sec\n', ...
        max(offsets));

    %% Store summary

    results(end+1).subject = subject;
    results(end).n_eeg = length(eeg_times);
    results(end).n_nirs = length(nirs_times);
    results(end).n_matched = n;
    results(end).class_matches = sum(class_match);
    results(end).median_offset = median(offsets);
    results(end).mean_offset = mean(offsets);
    results(end).min_offset = min(offsets);
    results(end).max_offset = max(offsets);

end

%% Final summary

fprintf('\n\n============================================\n');
fprintf(' FINAL SUMMARY\n');
fprintf('============================================\n\n');

fprintf('%-8s %-8s %-8s %-10s %-12s\n', ...
    'Subject','EEG','NIRS','ClassMatch','MedianOffset');

for k = 1:length(results)

    fprintf('%-8s %-8d %-8d %-10d %-12.3f\n', ...
        results(k).subject, ...
        results(k).n_eeg, ...
        results(k).n_nirs, ...
        results(k).class_matches, ...
        results(k).median_offset);

end

fprintf('\nDone.\n');