%% ============================================================
% FUSION PROJECT
% STEP 1 - EEG N-BACK PREPROCESSING
%
% Output:
%   X.mat
%   y.mat
%   subject_id.mat
%   trial_id.mat
%   metadata.csv
%   channel_names.json
%   label_map.json
%   preprocessing_config.json
%
% Classes:
%   0 = 0-back
%   1 = 2-back
%   2 = 3-back
%
% EEG input:
%   (N_epochs, N_channels, N_samples)
%
% =============================================================

clear;
clc;

%% ============================================================
% CONFIGURATION
% =============================================================

DATA_ROOT = 'D:\major_project_group50\dataset';

OUTPUT_ROOT = ...
    'D:\major_project_group50\project\FUSION\data\eeg';

if ~exist(OUTPUT_ROOT, 'dir')
    mkdir(OUTPUT_ROOT);
end

% EEG sampling frequency
FS = 200;

% Keep EEG only
% Original channels:
% 1:28  = EEG
% 29    = HEOG
% 30    = VEOG
EEG_CHANNELS = 1:28;

% ------------------------------------------------------------
% Epoch definition
% ------------------------------------------------------------
%
% We use a fixed-length epoch around each N-back stimulus.
%
% IMPORTANT:
% We keep this definition explicit in the configuration file
% so that the same methodological choice can be documented
% and reproduced.
%
% Current starting definition:
%
%   start = stimulus marker
%   duration = 2 seconds
%
% Therefore:
%   2 sec × 200 Hz = 400 samples
%
% ------------------------------------------------------------

EPOCH_START_SEC = 0.0;
EPOCH_END_SEC   = 2.0;

EPOCH_START_SAMPLES = round(EPOCH_START_SEC * FS);
EPOCH_END_SAMPLES   = round(EPOCH_END_SEC * FS);

N_SAMPLES = EPOCH_END_SAMPLES - EPOCH_START_SAMPLES;

fprintf('\n============================================\n');
fprintf('EEG N-BACK PREPROCESSING\n');
fprintf('============================================\n');

fprintf('Sampling rate : %d Hz\n', FS);
fprintf('EEG channels  : %d\n', length(EEG_CHANNELS));
fprintf('Epoch length  : %.2f sec\n', ...
    EPOCH_END_SEC - EPOCH_START_SEC);
fprintf('Samples/epoch : %d\n', N_SAMPLES);

%% ============================================================
% SUBJECTS
% ============================================================

% Fusion requires subjects with both EEG and fNIRS.
%
% VP012 and VP013 are EEG-only and therefore excluded.

SUBJECTS = { ...
    'VP001','VP002','VP003','VP004','VP005', ...
    'VP006','VP007','VP008','VP009','VP010','VP011', ...
    'VP014','VP015','VP016','VP017','VP018','VP019', ...
    'VP020','VP021','VP022','VP023','VP024','VP025','VP026' ...
};

%% ============================================================
% MARKER DEFINITIONS
% ============================================================

% Dataset marker classes
MARKER_0BACK_TARGET     = 1;
MARKER_2BACK_TARGET     = 2;
MARKER_2BACK_NONTARGET  = 3;
MARKER_3BACK_TARGET     = 4;
MARKER_3BACK_NONTARGET  = 5;

MARKER_0BACK_SESSION    = 6;
MARKER_2BACK_SESSION    = 7;
MARKER_3BACK_SESSION    = 8;

SESSION_MARKERS = [ ...
    MARKER_0BACK_SESSION, ...
    MARKER_2BACK_SESSION, ...
    MARKER_3BACK_SESSION];

STIMULUS_MARKERS = [ ...
    MARKER_0BACK_TARGET, ...
    MARKER_2BACK_TARGET, ...
    MARKER_2BACK_NONTARGET, ...
    MARKER_3BACK_TARGET, ...
    MARKER_3BACK_NONTARGET];

%% ============================================================
% STORAGE
% ============================================================

all_X = [];
all_y = {};

all_subject_id = {};
all_trial_id = [];

metadata_subject = {};
metadata_trial = [];
metadata_task = {};
metadata_label = [];

global_epoch_id = 0;

%% ============================================================
% PROCESS EACH SUBJECT
% ============================================================

for s = 1:length(SUBJECTS)

    subject = SUBJECTS{s};

    fprintf('\n--------------------------------------------\n');
    fprintf('Processing %s\n', subject);
    fprintf('--------------------------------------------\n');

    subject_folder = fullfile( ...
        DATA_ROOT, ...
        [subject '-EEG']);

    cnt_file = fullfile( ...
        subject_folder, 'cnt_nback.mat');

    mrk_file = fullfile( ...
        subject_folder, 'mrk_nback.mat');

    %% --------------------------------------------------------
    % Check files
    % --------------------------------------------------------

    if ~exist(cnt_file, 'file')
        warning('%s: cnt_nback.mat not found. Skipping.', subject);
        continue;
    end

    if ~exist(mrk_file, 'file')
        warning('%s: mrk_nback.mat not found. Skipping.', subject);
        continue;
    end

    %% --------------------------------------------------------
    % Load
    % --------------------------------------------------------

    cnt_struct = load(cnt_file);
    mrk_struct = load(mrk_file);

    cnt = cnt_struct.cnt_nback;
    mrk = mrk_struct.mrk_nback;

    %% --------------------------------------------------------
    % Basic validation
    % --------------------------------------------------------

    if cnt.fs ~= FS
        error( ...
            '%s: unexpected sampling rate %.2f Hz.', ...
            subject, cnt.fs);
    end

    X_raw = double(cnt.x(:, EEG_CHANNELS));

    fprintf('Raw EEG samples : %d\n', size(X_raw,1));
    fprintf('EEG channels     : %d\n', size(X_raw,2));
    fprintf('Markers          : %d\n', length(mrk.time));

    %% --------------------------------------------------------
    % Determine marker class for every event
    % --------------------------------------------------------

    [~, marker_class] = max(mrk.y, [], 1);

    marker_times = mrk.time;

    %% --------------------------------------------------------
    % Find session markers
    % --------------------------------------------------------

    session_indices = find( ...
        ismember(marker_class, SESSION_MARKERS));

    fprintf('Session markers : %d\n', length(session_indices));

    if length(session_indices) ~= 3
        warning( ...
            '%s: expected 3 N-back sessions, found %d.', ...
            subject, length(session_indices));
    end

    %% --------------------------------------------------------
    % Process stimulus markers
    % --------------------------------------------------------

    subject_epoch_count = 0;

    for e = 1:length(marker_times)

        current_marker = marker_class(e);

        % Ignore session markers
        if ~ismember(current_marker, STIMULUS_MARKERS)
            continue;
        end

        %% ----------------------------------------------------
        % Determine which session this stimulus belongs to
        % ----------------------------------------------------

        previous_sessions = ...
            session_indices(session_indices < e);

        if isempty(previous_sessions)
            warning( ...
                '%s: stimulus marker %d occurs before any session marker.', ...
                subject, e);
            continue;
        end

        latest_session_index = previous_sessions(end);

        session_marker_class = ...
            marker_class(latest_session_index);

        %% ----------------------------------------------------
        % Convert session to workload label
        % ----------------------------------------------------

        switch session_marker_class

            case MARKER_0BACK_SESSION

                label = 0;
                task = '0-back';

            case MARKER_2BACK_SESSION

                label = 1;
                task = '2-back';

            case MARKER_3BACK_SESSION

                label = 2;
                task = '3-back';

            otherwise

                warning( ...
                    '%s: unknown session marker.', ...
                    subject);
                continue;

        end

        %% ----------------------------------------------------
        % Epoch boundaries
        % ----------------------------------------------------

        marker_sample = round(marker_times(e));

        start_sample = ...
            marker_sample + EPOCH_START_SAMPLES;

        end_sample = ...
            marker_sample + EPOCH_END_SAMPLES - 1;

        %% ----------------------------------------------------
        % Boundary check
        % ----------------------------------------------------

        if start_sample < 1 || ...
                end_sample > size(X_raw,1)

            warning( ...
                '%s: epoch %d exceeds recording boundary. Skipping.', ...
                subject, e);

            continue;
        end

        %% ----------------------------------------------------
        % Extract epoch
        % ----------------------------------------------------

        epoch = X_raw( ...
            start_sample:end_sample, :);

        % Convert:
        %
        % current:
        %   samples × channels
        %
        % required:
        %   channels × samples

        epoch = epoch';

        %% ----------------------------------------------------
        % Store
        % ----------------------------------------------------

        all_X = cat(1, all_X, ...
            reshape(epoch, ...
            [1, size(epoch,1), size(epoch,2)]));

        all_y{end+1,1} = label;

        all_subject_id{end+1,1} = subject;

        % Original event/marker index
        all_trial_id(end+1,1) = e;

        metadata_subject{end+1,1} = subject;
        metadata_trial(end+1,1) = e;
        metadata_task{end+1,1} = task;
        metadata_label(end+1,1) = label;

        global_epoch_id = global_epoch_id + 1;
        subject_epoch_count = subject_epoch_count + 1;

    end

    fprintf( ...
        'Extracted epochs : %d\n', ...
        subject_epoch_count);

end

%% ============================================================
% CONVERT LABELS
% ============================================================

y = cell2mat(all_y);

subject_id = string(all_subject_id);

trial_id = all_trial_id;

%% ============================================================
% FINAL VALIDATION
% ============================================================

fprintf('\n============================================\n');
fprintf('FINAL EEG DATASET VALIDATION\n');
fprintf('============================================\n');

fprintf('X shape:\n');
disp(size(all_X));

fprintf('y shape:\n');
disp(size(y));

fprintf('subject_id shape:\n');
disp(size(subject_id));

fprintf('trial_id shape:\n');
disp(size(trial_id));

%% ------------------------------------------------------------
% Check dimensions
% ------------------------------------------------------------

N = size(all_X,1);

assert(size(all_X,2) == length(EEG_CHANNELS), ...
    'Incorrect EEG channel dimension.');

assert(size(all_X,3) == N_SAMPLES, ...
    'Incorrect epoch sample dimension.');

assert(length(y) == N, ...
    'y length does not match X.');

assert(length(subject_id) == N, ...
    'subject_id length does not match X.');

assert(length(trial_id) == N, ...
    'trial_id length does not match X.');

%% ------------------------------------------------------------
% Check labels
% ------------------------------------------------------------

assert(all(ismember(y, [0 1 2])), ...
    'Invalid class label detected.');

fprintf('\nClass counts:\n');

fprintf('0-back: %d\n', sum(y == 0));
fprintf('2-back: %d\n', sum(y == 1));
fprintf('3-back: %d\n', sum(y == 2));

%% ============================================================
% CREATE METADATA TABLE
% ============================================================

epoch_id = (1:N)';

metadata = table( ...
    epoch_id, ...
    subject_id, ...
    trial_id, ...
    string(metadata_task), ...
    metadata_label, ...
    'VariableNames', { ...
    'epoch_id', ...
    'subject_id', ...
    'trial_id', ...
    'task', ...
    'label'});

%% ============================================================
% CHANNEL NAMES
% ============================================================

channel_names = cnt.clab(EEG_CHANNELS);

%% ============================================================
% SAVE X
% ============================================================

X = all_X;

save( ...
    fullfile(OUTPUT_ROOT, 'X.mat'), ...
    'X', ...
    '-v7.3');

%% ============================================================
% SAVE LABELS
% ============================================================

save( ...
    fullfile(OUTPUT_ROOT, 'y.mat'), ...
    'y');

%% ============================================================
% SAVE SUBJECT IDS
% ============================================================

save( ...
    fullfile(OUTPUT_ROOT, 'subject_id.mat'), ...
    'subject_id');

%% ============================================================
% SAVE TRIAL IDS
% ============================================================

save( ...
    fullfile(OUTPUT_ROOT, 'trial_id.mat'), ...
    'trial_id');

%% ============================================================
% SAVE METADATA
% ============================================================

writetable( ...
    metadata, ...
    fullfile(OUTPUT_ROOT, 'metadata.csv'));

%% ============================================================
% SAVE CHANNEL NAMES
% ============================================================

channel_json = jsonencode(channel_names);

fid = fopen( ...
    fullfile(OUTPUT_ROOT, 'channel_names.json'), ...
    'w');

fprintf(fid, '%s', channel_json);

fclose(fid);

%% ============================================================
% SAVE LABEL MAP
% ============================================================

label_map = struct();

label_map.class_0 = '0-back';
label_map.class_1 = '2-back';
label_map.class_2 = '3-back';

fid = fopen( ...
    fullfile(OUTPUT_ROOT, 'label_map.json'), ...
    'w');

fprintf(fid, '%s', jsonencode(label_map));

fclose(fid);

%% ============================================================
% SAVE PREPROCESSING CONFIG
% ============================================================

config = struct();

config.dataset = 'TU Berlin simultaneous EEG-NIRS';
config.task = 'N-back';
config.sampling_rate_hz = FS;
config.input_channels = 28;
config.original_channels = 30;
config.excluded_channels = {'HEOG','VEOG'};

config.epoch_start_seconds = EPOCH_START_SEC;
config.epoch_end_seconds = EPOCH_END_SEC;
config.epoch_samples = N_SAMPLES;

config.label_0 = '0-back';
config.label_1 = '2-back';
config.label_2 = '3-back';

config.normalization = ...
    'No global normalization applied; training-only normalization will be performed during model training.';

config.trial_id_definition = ...
    'Original EEG N-back marker index.';

config.note = ...
    'Epoch extraction and exact preprocessing choices are documented here.';

fid = fopen( ...
    fullfile(OUTPUT_ROOT, ...
    'preprocessing_config.json'), ...
    'w');

fprintf(fid, '%s', jsonencode(config));

fclose(fid);

%% ============================================================
% SUMMARY
% ============================================================

fprintf('\n============================================\n');
fprintf('EEG PREPROCESSING COMPLETE\n');
fprintf('============================================\n');

fprintf('Output folder:\n%s\n\n', OUTPUT_ROOT);

fprintf('X shape = ');
disp(size(X));

fprintf('Total epochs = %d\n', N);

fprintf('0-back = %d\n', sum(y == 0));
fprintf('2-back = %d\n', sum(y == 1));
fprintf('3-back = %d\n', sum(y == 2));

fprintf('\nSaved files:\n');
fprintf('  X.mat\n');
fprintf('  y.mat\n');
fprintf('  subject_id.mat\n');
fprintf('  trial_id.mat\n');
fprintf('  metadata.csv\n');
fprintf('  channel_names.json\n');
fprintf('  label_map.json\n');
fprintf('  preprocessing_config.json\n');

fprintf('\n============================================\n');