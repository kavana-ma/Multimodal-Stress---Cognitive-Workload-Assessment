%% EXTRACT PAIRED EEG + fNIRS N-BACK EPOCHS
clear; clc;

%% ============================================================
% PATHS
% =============================================================
base = 'D:\major_project_group50\dataset';

output_dir = ...
    'D:\major_project_group50\project\FUSION\data\epochs';

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

%% ============================================================
% SUBJECTS
% =============================================================
subjects = { ...
    'VP001','VP002','VP003','VP004','VP005','VP006', ...
    'VP007','VP008','VP009','VP010','VP011', ...
    'VP014','VP015','VP016','VP017','VP018','VP019', ...
    'VP020','VP021','VP022','VP023','VP024','VP025','VP026'};

%% ============================================================
% PARAMETERS
% =============================================================
EEG_FS = 200;
NIRS_FS = 10;

WINDOW_SEC = 42;

EEG_SAMPLES  = EEG_FS  * WINDOW_SEC;   % 8400
NIRS_SAMPLES = NIRS_FS * WINDOW_SEC;   % 420

% EEG session marker codes
EEG_SESSION_CODES = [112 128 144];

% Corresponding labels
% 0-back = 0
% 2-back = 1
% 3-back = 2
EEG_LABELS = [0 1 2];

fprintf('\n============================================\n');
fprintf(' N-BACK EPOCH EXTRACTION\n');
fprintf('============================================\n');

fprintf('EEG epoch  : 28 x %d\n', EEG_SAMPLES);
fprintf('fNIRS epoch: 36 x %d x 2\n', NIRS_SAMPLES);
fprintf('Window     : %d seconds\n\n', WINDOW_SEC);

%% ============================================================
% PROCESS SUBJECTS
% =============================================================
for s = 1:length(subjects)

    subject = subjects{s};

    fprintf('\nProcessing %s...\n', subject);

    %% --------------------------------------------------------
    % LOAD EEG
    % ---------------------------------------------------------
    eeg_file = fullfile( ...
        base, ...
        [subject '-EEG'], ...
        'cnt_nback.mat');

    eeg_data = load(eeg_file);
    cnt_eeg = eeg_data.cnt_nback;

    eeg_signal = cnt_eeg.x(:,1:28);

    eeg_time = cnt_eeg.fs;

    %% --------------------------------------------------------
    % LOAD EEG MARKERS
    % ---------------------------------------------------------
    eeg_mrk_file = fullfile( ...
        base, ...
        [subject '-EEG'], ...
        'mrk_nback.mat');

    eeg_mrk_data = load(eeg_mrk_file);
    mrk_eeg = eeg_mrk_data.mrk_nback;

    %% --------------------------------------------------------
    % FIND EEG SESSION MARKERS
    % ---------------------------------------------------------
    eeg_session_idx = [];

    for k = 1:length(mrk_eeg.event.desc)

        if ismember( ...
                mrk_eeg.event.desc(k), ...
                EEG_SESSION_CODES)

            eeg_session_idx(end+1) = k;
        end

    end

    %% --------------------------------------------------------
    % LOAD fNIRS
    % ---------------------------------------------------------
    nirs_file = fullfile( ...
        base, ...
        [subject '-NIRS'], ...
        'cnt_nback.mat');

    nirs_data = load(nirs_file);
    cnt_nirs = nirs_data.cnt_nback;

    oxy   = cnt_nirs.oxy.x;
    deoxy = cnt_nirs.deoxy.x;

    %% --------------------------------------------------------
    % LOAD fNIRS MARKERS
    % ---------------------------------------------------------
    nirs_mrk_file = fullfile( ...
        base, ...
        [subject '-NIRS'], ...
        'mrk_nback.mat');

    nirs_mrk_data = load(nirs_mrk_file);
    mrk_nirs = nirs_mrk_data.mrk_nback;

    %% --------------------------------------------------------
    % FIND NIRS SESSION MARKERS
    % ---------------------------------------------------------
    nirs_session_idx = 1:length(mrk_nirs.event.desc);

    %% --------------------------------------------------------
    % CHECK BLOCK COUNT
    % ---------------------------------------------------------
    if length(eeg_session_idx) ~= 27

        error( ...
            '%s: EEG has %d blocks instead of 27.', ...
            subject, ...
            length(eeg_session_idx));

    end

    if length(nirs_session_idx) ~= 27

        error( ...
            '%s: fNIRS has %d blocks instead of 27.', ...
            subject, ...
            length(nirs_session_idx));

    end

    %% --------------------------------------------------------
    % PREALLOCATE
    % ---------------------------------------------------------
    eeg_epochs = zeros( ...
        27, ...
        28, ...
        EEG_SAMPLES, ...
        'single');

    fnirs_epochs = zeros( ...
        27, ...
        36, ...
        NIRS_SAMPLES, ...
        2, ...
        'single');

    labels = zeros(27,1);

    trial_id = (1:27)';

    subject_ids = repmat( ...
        string(subject), ...
        27, ...
        1);

    eeg_start_times = zeros(27,1);
    nirs_start_times = zeros(27,1);

    %% ========================================================
    % EXTRACT EACH BLOCK
    % =========================================================
    for b = 1:27

        %% ----------------------------------------------------
        % EEG BLOCK START
        % -----------------------------------------------------
        eeg_marker_idx = eeg_session_idx(b);

        eeg_marker_raw = ...
            mrk_eeg.time(eeg_marker_idx);

        % Marker timebase is 1000 Hz
        eeg_start_sec = eeg_marker_raw / 1000;

        eeg_start_sample = ...
            round(eeg_start_sec * EEG_FS) + 1;

        eeg_end_sample = ...
            eeg_start_sample + EEG_SAMPLES - 1;

        %% ----------------------------------------------------
        % NIRS BLOCK START
        % -----------------------------------------------------
        nirs_marker_idx = nirs_session_idx(b);

        nirs_marker_raw = ...
            mrk_nirs.time(nirs_marker_idx);

        % NIRS marker timebase is also 1000 Hz
        nirs_start_sec = nirs_marker_raw / 1000;

        nirs_start_sample = ...
            round(nirs_start_sec * NIRS_FS) + 1;

        nirs_end_sample = ...
            nirs_start_sample + NIRS_SAMPLES - 1;

        %% ----------------------------------------------------
        % BOUNDARY CHECK
        % -----------------------------------------------------
        if eeg_end_sample > size(eeg_signal,1)

            error( ...
                '%s block %d: EEG window exceeds signal.', ...
                subject, b);

        end

        if nirs_end_sample > size(oxy,1)

            error( ...
                '%s block %d: fNIRS window exceeds signal.', ...
                subject, b);

        end

        %% ----------------------------------------------------
        % EXTRACT EEG
        % -----------------------------------------------------
        eeg_epoch = ...
            eeg_signal( ...
                eeg_start_sample:eeg_end_sample, ...
                :)';

        %% ----------------------------------------------------
        % EXTRACT fNIRS
        % -----------------------------------------------------
        oxy_epoch = ...
            oxy( ...
                nirs_start_sample:nirs_end_sample, ...
                :)';

        deoxy_epoch = ...
            deoxy( ...
                nirs_start_sample:nirs_end_sample, ...
                :)';

        %% ----------------------------------------------------
        % STORE
        % -----------------------------------------------------
        eeg_epochs(b,:,:) = single(eeg_epoch);

        fnirs_epochs(b,:,:,1) = ...
            single(oxy_epoch);

        fnirs_epochs(b,:,:,2) = ...
            single(deoxy_epoch);

        %% ----------------------------------------------------
        % LABEL
        % -----------------------------------------------------
        session_code = ...
            mrk_eeg.event.desc(eeg_marker_idx);

        class_idx = ...
            find(EEG_SESSION_CODES == session_code,1);

        labels(b) = EEG_LABELS(class_idx);

        eeg_start_times(b) = eeg_start_sec;
        nirs_start_times(b) = nirs_start_sec;

    end

    %% ========================================================
    % SAVE SUBJECT
    % =========================================================
    output_file = fullfile( ...
        output_dir, ...
        [subject '_nback_epochs.mat']);

    save( ...
        output_file, ...
        'eeg_epochs', ...
        'fnirs_epochs', ...
        'labels', ...
        'trial_id', ...
        'subject_ids', ...
        'eeg_start_times', ...
        'nirs_start_times', ...
        '-v7.3');

    fprintf( ...
        '  Saved: %s\n', ...
        output_file);

end

fprintf('\n============================================\n');
fprintf('EXTRACTION COMPLETE\n');
fprintf('============================================\n');