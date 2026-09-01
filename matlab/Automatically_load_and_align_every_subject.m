%% ============================================================
% MULTI-SUBJECT EEG-fNIRS LOADING + TEMPORAL ALIGNMENT
% ============================================================

clear;
clc;

%% ============================================================
% DATASET LOCATION
% ============================================================

dataset_root = "D:\major_project_group50\dataset";

%% ============================================================
% FIND SUBJECT FOLDERS
% ============================================================

all_folders = dir(dataset_root);

EEG_subjects  = {};
NIRS_subjects = {};

for i = 1:length(all_folders)

    if ~all_folders(i).isdir
        continue;
    end

    folder_name = all_folders(i).name;

    if strcmp(folder_name,'.') || strcmp(folder_name,'..')
        continue;
    end

    % EEG
    token = regexp(folder_name, '^(VP\d+)-EEG$', 'tokens');

    if ~isempty(token)
        EEG_subjects{end+1} = token{1}{1};
    end

    % NIRS
    token = regexp(folder_name, '^(VP\d+)-NIRS$', 'tokens');

    if ~isempty(token)
        NIRS_subjects{end+1} = token{1}{1};
    end

end

%% ============================================================
% FIND COMMON SUBJECTS
% ============================================================

paired_subjects = intersect(EEG_subjects, NIRS_subjects);

subject_numbers = cellfun( ...
    @(x) str2double(extractAfter(x,2)), ...
    paired_subjects);

[~,order] = sort(subject_numbers);

paired_subjects = paired_subjects(order);

fprintf('\nFound %d paired subjects.\n', ...
    length(paired_subjects));

disp(paired_subjects');

%% ============================================================
% RESULT STORAGE
% ============================================================

results = struct();

%% ============================================================
% PROCESS EACH SUBJECT
% ============================================================

for s = 1:length(paired_subjects)

    subject = paired_subjects{s};

    fprintf('\n');
    fprintf('====================================================\n');
    fprintf('PROCESSING %s\n', subject);
    fprintf('====================================================\n');

    %% --------------------------------------------------------
    % FILE PATHS
    % ---------------------------------------------------------

    eeg_folder = fullfile( ...
        dataset_root, subject + "-EEG");

    nirs_folder = fullfile( ...
        dataset_root, subject + "-NIRS");

    eeg_cnt_file = fullfile( ...
        eeg_folder, "cnt_vf.mat");

    eeg_mrk_file = fullfile( ...
        eeg_folder, "mrk_vf.mat");

    nirs_cnt_file = fullfile( ...
        nirs_folder, "cnt_vf.mat");

    nirs_mrk_file = fullfile( ...
        nirs_folder, "mrk_vf.mat");

    %% --------------------------------------------------------
    % CHECK FILES
    % ---------------------------------------------------------

    if ~isfile(eeg_cnt_file) || ...
       ~isfile(eeg_mrk_file) || ...
       ~isfile(nirs_cnt_file) || ...
       ~isfile(nirs_mrk_file)

        fprintf('SKIPPED: missing file(s)\n');
        continue;
    end

    %% --------------------------------------------------------
    % LOAD EEG
    % ---------------------------------------------------------

    EEG_data = load(eeg_cnt_file);
    EEG_mrk_data = load(eeg_mrk_file);

    cnt_EEG = EEG_data.cnt_vf;
    mrk_EEG = EEG_mrk_data.mrk_vf;

    %% --------------------------------------------------------
    % LOAD NIRS
    % ---------------------------------------------------------

    NIRS_data = load(nirs_cnt_file);
    NIRS_mrk_data = load(nirs_mrk_file);

    cnt_NIRS = NIRS_data.cnt_vf;
    mrk_NIRS = NIRS_mrk_data.mrk_vf;

    %% --------------------------------------------------------
    % BASIC SIGNAL INFORMATION
    % ---------------------------------------------------------

    EEG_fs = cnt_EEG.fs;

    NIRS_oxy_fs   = cnt_NIRS.oxy.fs;
    NIRS_deoxy_fs = cnt_NIRS.deoxy.fs;

    EEG_duration = size(cnt_EEG.x,1) / EEG_fs;

    NIRS_duration = ...
        size(cnt_NIRS.oxy.x,1) / NIRS_oxy_fs;

    fprintf('EEG:  %.2f Hz, %.2f sec\n', ...
        EEG_fs, EEG_duration);

    fprintf('NIRS: %.2f Hz, %.2f sec\n', ...
        NIRS_oxy_fs, NIRS_duration);

    %% --------------------------------------------------------
    % MARKER TIMES
    % ---------------------------------------------------------

    EEG_time_sec = mrk_EEG.time / 1000;

    NIRS_time_sec = mrk_NIRS.time / 1000;

    nEEG  = length(EEG_time_sec);
    nNIRS = length(NIRS_time_sec);

    fprintf('EEG markers  = %d\n', nEEG);
    fprintf('NIRS markers = %d\n', nNIRS);

    %% --------------------------------------------------------
    % CHECK NUMBER OF EVENTS
    % ---------------------------------------------------------

    if nEEG ~= nNIRS

        fprintf('WARNING: event count mismatch!\n');
        continue;

    end

    nEvents = nEEG;

    %% --------------------------------------------------------
    % RAW MARKER DIFFERENCE
    % ---------------------------------------------------------

    delta = NIRS_time_sec - EEG_time_sec;

    fprintf('\nRaw timing difference:\n');

    fprintf('Mean   = %.4f sec\n', mean(delta));
    fprintf('Median = %.4f sec\n', median(delta));
    fprintf('Std    = %.4f sec\n', std(delta));
    fprintf('Min    = %.4f sec\n', min(delta));
    fprintf('Max    = %.4f sec\n', max(delta));

    %% --------------------------------------------------------
    % SEGMENT-WISE ALIGNMENT
    %
    % Current dataset structure:
    % 60 events
    % 1-20
    % 21-40
    % 41-60
    % ---------------------------------------------------------

    if nEvents == 60

        offset1 = median(delta(1:20));
        offset2 = median(delta(21:40));
        offset3 = median(delta(41:60));

        NIRS_aligned_time_sec = zeros(size(NIRS_time_sec));

        NIRS_aligned_time_sec(1:20) = ...
            NIRS_time_sec(1:20) - offset1;

        NIRS_aligned_time_sec(21:40) = ...
            NIRS_time_sec(21:40) - offset2;

        NIRS_aligned_time_sec(41:60) = ...
            NIRS_time_sec(41:60) - offset3;

        segment_offsets = ...
            [offset1 offset2 offset3];

    else

        % For subjects with a different number of events,
        % temporarily use one global median offset.

        fprintf(['WARNING: not 60 events. ' ...
                 'Using global median offset.\n']);

        global_offset = median(delta);

        NIRS_aligned_time_sec = ...
            NIRS_time_sec - global_offset;

        segment_offsets = global_offset;

    end

    %% --------------------------------------------------------
    % RESIDUAL
    % ---------------------------------------------------------

    residual = ...
        NIRS_aligned_time_sec - EEG_time_sec;

    fprintf('\nAfter alignment:\n');

    fprintf('Mean residual   = %.6f sec\n', ...
        mean(residual));

    fprintf('Median residual = %.6f sec\n', ...
        median(residual));

    fprintf('Std residual    = %.6f sec\n', ...
        std(residual));

    fprintf('Min residual    = %.6f sec\n', ...
        min(residual));

    fprintf('Max residual    = %.6f sec\n', ...
        max(residual));

    %% --------------------------------------------------------
    % CLASS AGREEMENT
    % ---------------------------------------------------------

    class_match = true;

    for e = 1:nEvents

        eeg_class_idx = ...
            find(mrk_EEG.y(:,e) ~= 0);

        nirs_class_idx = ...
            find(mrk_NIRS.y(:,e) ~= 0);

        if isempty(eeg_class_idx) || ...
           isempty(nirs_class_idx)

            class_match = false;
            continue;

        end

        eeg_class = ...
            mrk_EEG.className{eeg_class_idx(1)};

        nirs_class = ...
            mrk_NIRS.className{nirs_class_idx(1)};

        if ~strcmp(eeg_class,nirs_class)

            fprintf('CLASS MISMATCH: event %d\n',e);

            class_match = false;

        end

    end

    if class_match
        fprintf('Class agreement: YES\n');
    else
        fprintf('Class agreement: NO\n');
    end

    %% --------------------------------------------------------
    % STORE RESULTS
    % ---------------------------------------------------------

    results(s).subject = subject;

    results(s).EEG_fs = EEG_fs;

    results(s).NIRS_oxy_fs = NIRS_oxy_fs;

    results(s).NIRS_deoxy_fs = NIRS_deoxy_fs;

    results(s).EEG_duration = EEG_duration;

    results(s).NIRS_duration = NIRS_duration;

    results(s).nEvents = nEvents;

    results(s).EEG_time_sec = EEG_time_sec;

    results(s).NIRS_time_sec = NIRS_time_sec;

    results(s).NIRS_aligned_time_sec = ...
        NIRS_aligned_time_sec;

    results(s).residual = residual;

    results(s).segment_offsets = ...
        segment_offsets;

    results(s).class_match = class_match;

    results(s).cnt_EEG = cnt_EEG;

    results(s).cnt_NIRS = cnt_NIRS;

    results(s).mrk_EEG = mrk_EEG;

    results(s).mrk_NIRS = mrk_NIRS;

end