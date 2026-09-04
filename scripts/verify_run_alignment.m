%% VERIFY THREE-RUN EEG-fNIRS ALIGNMENT
%
% Checks the temporal offset separately for the three apparent
% recording runs.
%
% NO preprocessing.
% NO epoch extraction.
% NO data modification.

clear;
clc;

base = 'D:\major_project_group50\dataset';

subjects = { ...
    'VP001','VP002','VP003','VP004','VP005','VP006', ...
    'VP007','VP008','VP009','VP010','VP011', ...
    'VP014','VP015','VP016','VP017','VP018','VP019', ...
    'VP020','VP021','VP022','VP023','VP024','VP025','VP026'};

eeg_codes  = [112 128 144];
nirs_codes = [7 8 9];

class_labels = [0 2 3];

fprintf('\n');
fprintf('============================================================\n');
fprintf(' THREE-RUN EEG-fNIRS ALIGNMENT VERIFICATION\n');
fprintf('============================================================\n');

for s = 1:length(subjects)

    subject = subjects{s};

    eeg_file = fullfile(base,[subject '-EEG'],'mrk_nback.mat');
    nirs_file = fullfile(base,[subject '-NIRS'],'mrk_nback.mat');

    eeg_data = load(eeg_file);
    nirs_data = load(nirs_file);

    eeg_mrk = eeg_data.mrk_nback;
    nirs_mrk = nirs_data.mrk_nback;

    %% Extract EEG session markers

    eeg_times = [];
    eeg_class = [];

    for k = 1:length(eeg_mrk.time)

        code = eeg_mrk.event.desc(k);

        idx = find(eeg_codes == code,1);

        if ~isempty(idx)
            eeg_times(end+1,1) = eeg_mrk.time(k)/1000;
            eeg_class(end+1,1) = class_labels(idx);
        end
    end

    %% Extract NIRS session markers

    nirs_times = [];
    nirs_class = [];

    for k = 1:length(nirs_mrk.time)

        code = nirs_mrk.event.desc(k);

        idx = find(nirs_codes == code,1);

        if ~isempty(idx)
            nirs_times(end+1,1) = nirs_mrk.time(k)/1000;
            nirs_class(end+1,1) = class_labels(idx);
        end
    end

    %% Verify

    n = min(length(eeg_times),length(nirs_times));

    offsets = nirs_times(1:n) - eeg_times(1:n);

    matches = eeg_class(1:n) == nirs_class(1:n);

    %% Three runs

    run1 = 1:9;
    run2 = 10:18;
    run3 = 19:27;

    fprintf('\n------------------------------------------------------------\n');
    fprintf('%s\n',subject);
    fprintf('------------------------------------------------------------\n');

    fprintf('Blocks EEG  : %d\n',length(eeg_times));
    fprintf('Blocks NIRS : %d\n',length(nirs_times));
    fprintf('Class matches: %d/%d\n',sum(matches),n);

    fprintf('\nRUN 1 (blocks 1-9)\n');
    fprintf('  Median offset = %.3f sec\n',median(offsets(run1)));
    fprintf('  Mean offset   = %.3f sec\n',mean(offsets(run1)));
    fprintf('  Range         = %.3f to %.3f sec\n', ...
        min(offsets(run1)),max(offsets(run1)));

    fprintf('\nRUN 2 (blocks 10-18)\n');
    fprintf('  Median offset = %.3f sec\n',median(offsets(run2)));
    fprintf('  Mean offset   = %.3f sec\n',mean(offsets(run2)));
    fprintf('  Range         = %.3f to %.3f sec\n', ...
        min(offsets(run2)),max(offsets(run2)));

    fprintf('\nRUN 3 (blocks 19-27)\n');
    fprintf('  Median offset = %.3f sec\n',median(offsets(run3)));
    fprintf('  Mean offset   = %.3f sec\n',mean(offsets(run3)));
    fprintf('  Range         = %.3f to %.3f sec\n', ...
        min(offsets(run3)),max(offsets(run3)));

end

fprintf('\n============================================================\n');
fprintf('DONE\n');
fprintf('============================================================\n');