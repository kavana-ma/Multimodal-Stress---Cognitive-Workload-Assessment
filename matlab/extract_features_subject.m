%% ============================================================
% EXTRACT_FEATURES_SUBJECT
%
% Extract EEG + fNIRS features for one subject
%
% Input:
%   preprocessed_epochs\VPxxx_preprocessed.mat
%
% Output:
%   features\VPxxx_features.mat
%
% Feature dimensions:
%
% EEG  = 28 channels x 10 features = 280
% HbO  = 36 channels x  6 features = 216
% HbR  = 36 channels x  6 features = 216
%
% TOTAL = 712 features / event
%
% Number of events = 60
%
% Final feature matrix:
%   60 x 712
%
% ============================================================

clear;
clc;

fprintf('\n============================================\n');
fprintf('SUBJECT FEATURE EXTRACTION\n');
fprintf('============================================\n');

%% ============================================================
% USER SETTINGS
% ============================================================

subject = 'VP001';

input_folder = ...
    'D:\major_project_group50\dataset\preprocessed_epochs';

output_folder = ...
    'D:\major_project_group50\dataset\features';

%% ============================================================
% CREATE OUTPUT FOLDER
% ============================================================

if ~isfolder(output_folder)
    mkdir(output_folder);
end

%% ============================================================
% INPUT FILE
% ============================================================

input_file = fullfile( ...
    input_folder, ...
    [subject '_preprocessed.mat']);

fprintf('\nLoading:\n%s\n', input_file);

if ~isfile(input_file)
    error('Input file not found: %s', input_file);
end

load(input_file);

%% ============================================================
% CHECK REQUIRED VARIABLES
% ============================================================

required_vars = { ...
    'EEG_preprocessed', ...
    'Oxy_preprocessed', ...
    'Deoxy_preprocessed', ...
    'eeg_fs', ...
    'nirs_fs', ...
    'eventClass'};

for k = 1:length(required_vars)

    if ~exist(required_vars{k},'var')

        error('Required variable missing: %s', ...
            required_vars{k});

    end

end

%% ============================================================
% INPUT INFORMATION
% ============================================================

[nEEG_samples,nEEG_channels,nEvents] = ...
    size(EEG_preprocessed);

[nNIRS_samples,nNIRS_channels,nNIRS_events] = ...
    size(Oxy_preprocessed);

[~,nDeoxy_channels,nDeoxy_events] = ...
    size(Deoxy_preprocessed);

fprintf('\n========== INPUT DATA ==========\n');

fprintf('EEG   = %d x %d x %d\n', ...
    nEEG_samples, ...
    nEEG_channels, ...
    nEvents);

fprintf('Oxy   = %d x %d x %d\n', ...
    nNIRS_samples, ...
    nNIRS_channels, ...
    nNIRS_events);

fprintf('Deoxy = %d x %d x %d\n', ...
    size(Deoxy_preprocessed,1), ...
    nDeoxy_channels, ...
    nDeoxy_events);

fprintf('EEG fs  = %.2f Hz\n', eeg_fs);
fprintf('NIRS fs = %.2f Hz\n', nirs_fs);

fprintf('Events = %d\n', nEvents);

%% ============================================================
% CHECK EVENT COUNTS
% ============================================================

if nEvents ~= nNIRS_events || ...
   nEvents ~= nDeoxy_events

    error('EEG/NIRS event count mismatch.');

end

%% ============================================================
% CLASS COUNTS
% ============================================================

BL_idx = eventClass == "BL";
VF_idx = eventClass == "VF";

fprintf('\n========== CLASS COUNTS ==========\n');

fprintf('BL = %d\n', sum(BL_idx));
fprintf('VF = %d\n', sum(VF_idx));

if sum(BL_idx) ~= 30 || sum(VF_idx) ~= 30

    warning(['Expected 30 BL and 30 VF trials. ' ...
             'Check event labels.']);

end

%% ============================================================
% FEATURE CONFIGURATION
% ============================================================

EEG_features_per_channel   = 10;
HbO_features_per_channel   = 6;
HbR_features_per_channel   = 6;

EEG_total_features = ...
    nEEG_channels * EEG_features_per_channel;

HbO_total_features = ...
    nNIRS_channels * HbO_features_per_channel;

HbR_total_features = ...
    nDeoxy_channels * HbR_features_per_channel;

total_features = ...
    EEG_total_features + ...
    HbO_total_features + ...
    HbR_total_features;

fprintf('\n========== FEATURE CONFIGURATION ==========\n');

fprintf('EEG channels       = %d\n', nEEG_channels);
fprintf('EEG features/ch    = %d\n', ...
    EEG_features_per_channel);

fprintf('HbO channels       = %d\n', nNIRS_channels);
fprintf('HbO features/ch    = %d\n', ...
    HbO_features_per_channel);

fprintf('HbR channels       = %d\n', nDeoxy_channels);
fprintf('HbR features/ch    = %d\n', ...
    HbR_features_per_channel);

fprintf('Total features     = %d\n', ...
    total_features);

%% ============================================================
% EXPECTED TOTAL
% ============================================================

expected_features = 712;

if total_features ~= expected_features

    warning(['Current configuration produces %d features, ' ...
             'not 712.'], total_features);

end

%% ============================================================
% PREALLOCATE FEATURE MATRIX
% ============================================================

FeatureMatrix = zeros(nEvents,total_features);

%% ============================================================
% FEATURE NAMES
% ============================================================

FeatureNames = strings(1,total_features);

feature_counter = 1;

%% ============================================================
% EEG FEATURE NAMES
% ============================================================

EEG_feature_names = [ ...
    "Mean", ...
    "Std", ...
    "Variance", ...
    "RMS", ...
    "PeakToPeak", ...
    "Maximum", ...
    "Minimum", ...
    "HjorthActivity", ...
    "HjorthMobility", ...
    "HjorthComplexity"];

for ch = 1:nEEG_channels

    for f = 1:EEG_features_per_channel

        FeatureNames(feature_counter) = ...
            "EEG_Ch" + ch + "_" + EEG_feature_names(f);

        feature_counter = feature_counter + 1;

    end

end

%% ============================================================
% HbO FEATURE NAMES
% ============================================================

NIRS_feature_names = [ ...
    "Mean", ...
    "Std", ...
    "Maximum", ...
    "Minimum", ...
    "Range", ...
    "AUC"];

for ch = 1:nNIRS_channels

    for f = 1:HbO_features_per_channel

        FeatureNames(feature_counter) = ...
            "HbO_Ch" + ch + "_" + NIRS_feature_names(f);

        feature_counter = feature_counter + 1;

    end

end

%% ============================================================
% HbR FEATURE NAMES
% ============================================================

for ch = 1:nDeoxy_channels

    for f = 1:HbR_features_per_channel

        FeatureNames(feature_counter) = ...
            "HbR_Ch" + ch + "_" + NIRS_feature_names(f);

        feature_counter = feature_counter + 1;

    end

end

%% ============================================================
% VERIFY FEATURE NAME COUNT
% ============================================================

if feature_counter - 1 ~= total_features

    error('Feature name count mismatch.');

end

%% ============================================================
% FEATURE EXTRACTION
% ============================================================

fprintf('\n========== EXTRACTING ==========\n');

for event = 1:nEvents

    fprintf('Event %d / %d\n',event,nEvents);

    feature_counter = 1;

    %% ========================================================
    % EEG FEATURES
    % ========================================================

    for ch = 1:nEEG_channels

        signal = EEG_preprocessed(:,ch,event);

        % Remove NaN/Inf if present
        signal = signal(isfinite(signal));

        if isempty(signal)

            error(['Empty EEG signal at event %d, ' ...
                   'channel %d.'], ...
                   event,ch);

        end

        %% Feature 1: Mean

        f_mean = mean(signal);

        %% Feature 2: Standard deviation

        f_std = std(signal);

        %% Feature 3: Variance

        f_variance = var(signal);

        %% Feature 4: RMS

        f_rms = sqrt(mean(signal.^2));

        %% Feature 5: Peak-to-peak

        f_ptp = max(signal) - min(signal);

        %% Feature 6: Maximum

        f_max = max(signal);

        %% Feature 7: Minimum

        f_min = min(signal);

        %% Feature 8: Hjorth Activity

        f_activity = var(signal);

        %% Feature 9: Hjorth Mobility

        dx = diff(signal);

        if std(signal) == 0

            f_mobility = 0;

        else

            f_mobility = ...
                std(dx) / std(signal);

        end

        %% Feature 10: Hjorth Complexity

        ddx = diff(dx);

        if std(dx) == 0 || std(signal) == 0

            f_complexity = 0;

        else

            mobility_dx = ...
                std(ddx) / std(dx);

            f_complexity = ...
                mobility_dx / f_mobility;

        end

        %% Store EEG features

        FeatureMatrix(event,feature_counter) = f_mean;
        feature_counter = feature_counter + 1;

        FeatureMatrix(event,feature_counter) = f_std;
        feature_counter = feature_counter + 1;

        FeatureMatrix(event,feature_counter) = f_variance;
        feature_counter = feature_counter + 1;

        FeatureMatrix(event,feature_counter) = f_rms;
        feature_counter = feature_counter + 1;

        FeatureMatrix(event,feature_counter) = f_ptp;
        feature_counter = feature_counter + 1;

        FeatureMatrix(event,feature_counter) = f_max;
        feature_counter = feature_counter + 1;

        FeatureMatrix(event,feature_counter) = f_min;
        feature_counter = feature_counter + 1;

        FeatureMatrix(event,feature_counter) = f_activity;
        feature_counter = feature_counter + 1;

        FeatureMatrix(event,feature_counter) = f_mobility;
        feature_counter = feature_counter + 1;

        FeatureMatrix(event,feature_counter) = f_complexity;
        feature_counter = feature_counter + 1;

    end

    %% ========================================================
    % HbO FEATURES
    % ========================================================

    for ch = 1:nNIRS_channels

        signal = Oxy_preprocessed(:,ch,event);

        signal = signal(isfinite(signal));

        if isempty(signal)

            error(['Empty HbO signal at event %d, ' ...
                   'channel %d.'], ...
                   event,ch);

        end

        %% Mean

        f_mean = mean(signal);

        %% Standard deviation

        f_std = std(signal);

        %% Maximum

        f_max = max(signal);

        %% Minimum

        f_min = min(signal);

        %% Range

        f_range = f_max - f_min;

        %% AUC

        f_auc = trapz(signal) / nirs_fs;

        %% Store HbO features

        FeatureMatrix(event,feature_counter) = f_mean;
        feature_counter = feature_counter + 1;

        FeatureMatrix(event,feature_counter) = f_std;
        feature_counter = feature_counter + 1;

        FeatureMatrix(event,feature_counter) = f_max;
        feature_counter = feature_counter + 1;

        FeatureMatrix(event,feature_counter) = f_min;
        feature_counter = feature_counter + 1;

        FeatureMatrix(event,feature_counter) = f_range;
        feature_counter = feature_counter + 1;

        FeatureMatrix(event,feature_counter) = f_auc;
        feature_counter = feature_counter + 1;

    end

    %% ========================================================
    % HbR FEATURES
    % ========================================================

    for ch = 1:nDeoxy_channels

        signal = Deoxy_preprocessed(:,ch,event);

        signal = signal(isfinite(signal));

        if isempty(signal)

            error(['Empty HbR signal at event %d, ' ...
                   'channel %d.'], ...
                   event,ch);

        end

        %% Mean

        f_mean = mean(signal);

        %% Standard deviation

        f_std = std(signal);

        %% Maximum

        f_max = max(signal);

        %% Minimum

        f_min = min(signal);

        %% Range

        f_range = f_max - f_min;

        %% AUC

        f_auc = trapz(signal) / nirs_fs;

        %% Store HbR features

        FeatureMatrix(event,feature_counter) = f_mean;
        feature_counter = feature_counter + 1;

        FeatureMatrix(event,feature_counter) = f_std;
        feature_counter = feature_counter + 1;

        FeatureMatrix(event,feature_counter) = f_max;
        feature_counter = feature_counter + 1;

        FeatureMatrix(event,feature_counter) = f_min;
        feature_counter = feature_counter + 1;

        FeatureMatrix(event,feature_counter) = f_range;
        feature_counter = feature_counter + 1;

        FeatureMatrix(event,feature_counter) = f_auc;
        feature_counter = feature_counter + 1;

    end

    %% ========================================================
    % VERIFY EVENT FEATURE COUNT
    % ========================================================

    if feature_counter - 1 ~= total_features

        error(['Event %d feature count mismatch. ' ...
               'Expected %d, got %d.'], ...
               event, ...
               total_features, ...
               feature_counter - 1);

    end

end

%% ============================================================
% FEATURE MATRIX CHECK
% ============================================================

fprintf('\n========== FEATURE MATRIX ==========\n');

fprintf('Size = %d x %d\n', ...
    size(FeatureMatrix,1), ...
    size(FeatureMatrix,2));

%% ============================================================
% FINAL FEATURE COUNT CHECK
% ============================================================

if size(FeatureMatrix,2) ~= expected_features

    error(['Feature count mismatch. Expected %d, got %d.'], ...
        expected_features, ...
        size(FeatureMatrix,2));

end

fprintf('Feature count = %d\n', ...
    size(FeatureMatrix,2));

%% ============================================================
% CHECK NaN / INF
% ============================================================

fprintf('\n========== FEATURE QUALITY ==========\n');

fprintf('NaN count = %d\n', ...
    sum(isnan(FeatureMatrix(:))));

fprintf('Inf count = %d\n', ...
    sum(isinf(FeatureMatrix(:))));

if any(isnan(FeatureMatrix(:))) || ...
   any(isinf(FeatureMatrix(:)))

    error('Feature matrix contains NaN or Inf.');

end

%% ============================================================
% CLASS LABELS
% ============================================================

Labels = eventClass;

fprintf('\n========== LABELS ==========\n');

fprintf('BL = %d\n',sum(Labels == "BL"));
fprintf('VF = %d\n',sum(Labels == "VF"));

%% ============================================================
% SAVE
% ============================================================

output_file = fullfile( ...
    output_folder, ...
    [subject '_features.mat']);

save(output_file, ...
    'FeatureMatrix', ...
    'FeatureNames', ...
    'Labels', ...
    'SubjectID', ...
    'eeg_fs', ...
    'nirs_fs', ...
    '-v7.3');

fprintf('\n============================================\n');
fprintf('FEATURE EXTRACTION COMPLETE\n');
fprintf('============================================\n');

fprintf('Subject: %s\n',subject);

fprintf('Feature matrix: %d x %d\n', ...
    size(FeatureMatrix,1), ...
    size(FeatureMatrix,2));

fprintf('BL trials = %d\n',sum(Labels == "BL"));
fprintf('VF trials = %d\n',sum(Labels == "VF"));

fprintf('\nSaved:\n%s\n',output_file);

fprintf('============================================\n');