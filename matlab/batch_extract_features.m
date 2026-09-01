%% ============================================================
% BATCH FEATURE EXTRACTION
%
% Extract 712 features for all 24 paired subjects
%
% Input:
%   D:\major_project_group50\dataset\preprocessed_epochs
%
% Output:
%   D:\major_project_group50\dataset\features
%
% Each subject:
%   60 trials x 712 features
%   30 BL
%   30 VF
%
% ============================================================

clear;
clc;

fprintf('\n');
fprintf('============================================\n');
fprintf('BATCH FEATURE EXTRACTION\n');
fprintf('============================================\n');

%% ============================================================
% PATHS
% ============================================================

input_folder = ...
    'D:\major_project_group50\dataset\preprocessed_epochs';

output_folder = ...
    'D:\major_project_group50\dataset\features';

if ~isfolder(output_folder)
    mkdir(output_folder);
end

%% ============================================================
% FIND PREPROCESSED SUBJECT FILES
% ============================================================

files = dir(fullfile( ...
    input_folder, ...
    'VP*_preprocessed.mat'));

fprintf('\nPreprocessed subject files found = %d\n', ...
    length(files));

if isempty(files)

    error('No preprocessed subject files found.');

end

%% ============================================================
% SORT FILES BY SUBJECT NUMBER
% ============================================================

subject_numbers = zeros(length(files),1);

for k = 1:length(files)

    token = regexp( ...
        files(k).name, ...
        'VP(\d+)_preprocessed\.mat', ...
        'tokens');

    if isempty(token)

        error('Unexpected filename: %s', ...
            files(k).name);

    end

    subject_numbers(k) = str2double(token{1}{1});

end

[~,sort_idx] = sort(subject_numbers);

files = files(sort_idx);

%% ============================================================
% EXPECTED SUBJECT COUNT
% ============================================================

fprintf('\n============================================\n');
fprintf('SUBJECTS TO PROCESS\n');
fprintf('============================================\n');

for k = 1:length(files)

    fprintf('%d. VP%03d\n', ...
        k, ...
        subject_numbers(sort_idx(k)));

end

%% ============================================================
% PROCESS EACH SUBJECT
% ============================================================

success_count = 0;
failed_count = 0;

results = cell(length(files),6);

for k = 1:length(files)

    filename = files(k).name;

    token = regexp( ...
        filename, ...
        'VP(\d+)_preprocessed\.mat', ...
        'tokens');

    subject = ['VP' token{1}{1}];

    fprintf('\n\n');
    fprintf('============================================\n');
    fprintf('SUBJECT %s (%d/%d)\n', ...
        subject,k,length(files));
    fprintf('============================================\n');

    try

        %% ----------------------------------------------------
        % LOAD DATA
        % ----------------------------------------------------

        input_file = fullfile( ...
            input_folder, ...
            filename);

        load(input_file);

        %% ----------------------------------------------------
        % CHECK VARIABLES
        % ----------------------------------------------------

        required_vars = { ...
            'EEG_preprocessed', ...
            'Oxy_preprocessed', ...
            'Deoxy_preprocessed', ...
            'eeg_fs', ...
            'nirs_fs', ...
            'eventClass'};

        for v = 1:length(required_vars)

            if ~exist(required_vars{v},'var')

                error('Missing variable: %s', ...
                    required_vars{v});

            end

        end

        %% ----------------------------------------------------
        % DIMENSIONS
        % ----------------------------------------------------

        [nEEG_samples,nEEG_channels,nEvents] = ...
            size(EEG_preprocessed);

        [nNIRS_samples,nNIRS_channels,nNIRS_events] = ...
            size(Oxy_preprocessed);

        [~,nDeoxy_channels,nDeoxy_events] = ...
            size(Deoxy_preprocessed);

        %% ----------------------------------------------------
        % CHECK DIMENSIONS
        % ----------------------------------------------------

        if nEvents ~= 60
            error('Expected 60 events, got %d.',nEvents);
        end

        if nNIRS_events ~= nEvents || ...
           nDeoxy_events ~= nEvents

            error('EEG/NIRS event mismatch.');

        end

        if nEEG_channels ~= 28

            error('Expected 28 EEG channels, got %d.', ...
                nEEG_channels);

        end

        if nNIRS_channels ~= 36

            error('Expected 36 HbO channels, got %d.', ...
                nNIRS_channels);

        end

        if nDeoxy_channels ~= 36

            error('Expected 36 HbR channels, got %d.', ...
                nDeoxy_channels);

        end

        %% ----------------------------------------------------
        % CLASS CHECK
        % ----------------------------------------------------

        BL_idx = eventClass == "BL";
        VF_idx = eventClass == "VF";

        nBL = sum(BL_idx);
        nVF = sum(VF_idx);

        if nBL ~= 30 || nVF ~= 30

            error(['Expected 30 BL / 30 VF. ' ...
                   'Got %d BL / %d VF.'], ...
                   nBL,nVF);

        end

        %% ----------------------------------------------------
        % FEATURE CONFIGURATION
        % ----------------------------------------------------

        EEG_features_per_channel = 10;
        HbO_features_per_channel = 6;
        HbR_features_per_channel = 6;

        total_features = ...
            nEEG_channels * EEG_features_per_channel + ...
            nNIRS_channels * HbO_features_per_channel + ...
            nDeoxy_channels * HbR_features_per_channel;

        if total_features ~= 712

            error('Expected 712 features, got %d.', ...
                total_features);

        end

        %% ----------------------------------------------------
        % PREALLOCATE
        % ----------------------------------------------------

        FeatureMatrix = zeros(nEvents,total_features);

        %% ----------------------------------------------------
        % FEATURE EXTRACTION
        % ----------------------------------------------------

        for event = 1:nEvents

            feature_counter = 1;

            %% ================================================
            % EEG
            % ================================================

            for ch = 1:nEEG_channels

                signal = EEG_preprocessed(:,ch,event);

                signal = signal(isfinite(signal));

                if isempty(signal)

                    error(['Empty EEG signal: event %d ' ...
                           'channel %d.'], ...
                           event,ch);

                end

                f_mean = mean(signal);
                f_std  = std(signal);
                f_var  = var(signal);
                f_rms  = sqrt(mean(signal.^2));
                f_ptp  = max(signal) - min(signal);
                f_max  = max(signal);
                f_min  = min(signal);

                % Hjorth Activity
                f_activity = var(signal);

                % Hjorth Mobility
                dx = diff(signal);

                if std(signal) == 0
                    f_mobility = 0;
                else
                    f_mobility = ...
                        std(dx) / std(signal);
                end

                % Hjorth Complexity
                ddx = diff(dx);

                if std(dx) == 0 || ...
                   f_mobility == 0

                    f_complexity = 0;

                else

                    mobility_dx = ...
                        std(ddx) / std(dx);

                    f_complexity = ...
                        mobility_dx / f_mobility;

                end

                values = [ ...
                    f_mean ...
                    f_std ...
                    f_var ...
                    f_rms ...
                    f_ptp ...
                    f_max ...
                    f_min ...
                    f_activity ...
                    f_mobility ...
                    f_complexity];

                FeatureMatrix( ...
                    event, ...
                    feature_counter:feature_counter+9) = ...
                    values;

                feature_counter = ...
                    feature_counter + 10;

            end

            %% ================================================
            % HbO
            % ================================================

            for ch = 1:nNIRS_channels

                signal = Oxy_preprocessed(:,ch,event);

                signal = signal(isfinite(signal));

                if isempty(signal)

                    error(['Empty HbO signal: event %d ' ...
                           'channel %d.'], ...
                           event,ch);

                end

                f_mean = mean(signal);
                f_std  = std(signal);
                f_max  = max(signal);
                f_min  = min(signal);
                f_range = f_max - f_min;

                f_auc = trapz(signal) / nirs_fs;

                values = [ ...
                    f_mean ...
                    f_std ...
                    f_max ...
                    f_min ...
                    f_range ...
                    f_auc];

                FeatureMatrix( ...
                    event, ...
                    feature_counter:feature_counter+5) = ...
                    values;

                feature_counter = ...
                    feature_counter + 6;

            end

            %% ================================================
            % HbR
            % ================================================

            for ch = 1:nDeoxy_channels

                signal = Deoxy_preprocessed(:,ch,event);

                signal = signal(isfinite(signal));

                if isempty(signal)

                    error(['Empty HbR signal: event %d ' ...
                           'channel %d.'], ...
                           event,ch);

                end

                f_mean = mean(signal);
                f_std  = std(signal);
                f_max  = max(signal);
                f_min  = min(signal);
                f_range = f_max - f_min;

                f_auc = trapz(signal) / nirs_fs;

                values = [ ...
                    f_mean ...
                    f_std ...
                    f_max ...
                    f_min ...
                    f_range ...
                    f_auc];

                FeatureMatrix( ...
                    event, ...
                    feature_counter:feature_counter+5) = ...
                    values;

                feature_counter = ...
                    feature_counter + 6;

            end

            %% ------------------------------------------------
            % VERIFY
            % ------------------------------------------------

            if feature_counter - 1 ~= 712

                error(['Event %d: expected 712 features, ' ...
                       'got %d.'], ...
                       event, ...
                       feature_counter - 1);

            end

        end

        %% ----------------------------------------------------
        % QUALITY CHECK
        % ----------------------------------------------------

        nan_count = sum(isnan(FeatureMatrix(:)));
        inf_count = sum(isinf(FeatureMatrix(:)));

        if nan_count ~= 0 || inf_count ~= 0

            error(['Feature quality failure: ' ...
                   'NaN=%d, Inf=%d'], ...
                   nan_count,inf_count);

        end

        %% ----------------------------------------------------
        % FEATURE NAMES
        % ----------------------------------------------------

        FeatureNames = strings(1,712);

        counter = 1;

        EEG_names = [ ...
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

        NIRS_names = [ ...
            "Mean", ...
            "Std", ...
            "Maximum", ...
            "Minimum", ...
            "Range", ...
            "AUC"];

        for ch = 1:28

            for f = 1:10

                FeatureNames(counter) = ...
                    "EEG_Ch" + ch + "_" + EEG_names(f);

                counter = counter + 1;

            end

        end

        for ch = 1:36

            for f = 1:6

                FeatureNames(counter) = ...
                    "HbO_Ch" + ch + "_" + NIRS_names(f);

                counter = counter + 1;

            end

        end

        for ch = 1:36

            for f = 1:6

                FeatureNames(counter) = ...
                    "HbR_Ch" + ch + "_" + NIRS_names(f);

                counter = counter + 1;

            end

        end

        %% ----------------------------------------------------
        % SAVE
        % ----------------------------------------------------

        Labels = eventClass;
        SubjectID = subject;

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

        %% ----------------------------------------------------
        % RESULT
        % ----------------------------------------------------

        fprintf('\nSUCCESS\n');

        fprintf('Feature matrix = %d x %d\n', ...
            size(FeatureMatrix,1), ...
            size(FeatureMatrix,2));

        fprintf('BL = %d\n',nBL);
        fprintf('VF = %d\n',nVF);

        fprintf('NaN = %d\n',nan_count);
        fprintf('Inf = %d\n',inf_count);

        fprintf('Saved:\n%s\n',output_file);

        %% ----------------------------------------------------
        % STORE SUMMARY
        % ----------------------------------------------------

        success_count = success_count + 1;

        results{k,1} = subject;
        results{k,2} = 'SUCCESS';
        results{k,3} = size(FeatureMatrix,1);
        results{k,4} = size(FeatureMatrix,2);
        results{k,5} = nBL;
        results{k,6} = nVF;

    catch ME

        failed_count = failed_count + 1;

        fprintf('\nFAILED: %s\n',subject);
        fprintf('Reason: %s\n',ME.message);

        results{k,1} = subject;
        results{k,2} = 'FAILED';
        results{k,3} = NaN;
        results{k,4} = NaN;
        results{k,5} = NaN;
        results{k,6} = NaN;

    end

end

%% ============================================================
% BATCH SUMMARY
% ============================================================

fprintf('\n\n');
fprintf('============================================\n');
fprintf('BATCH FEATURE EXTRACTION COMPLETE\n');
fprintf('============================================\n');

fprintf('Subjects found  = %d\n',length(files));
fprintf('Successful      = %d\n',success_count);
fprintf('Failed          = %d\n',failed_count);

fprintf('\n============================================\n');
fprintf('SUBJECT SUMMARY\n');
fprintf('============================================\n');

fprintf('Subject\tStatus\t\tRows\tFeatures\tBL\tVF\n');

for k = 1:length(files)

    fprintf('%s\t%s\t%d\t%d\t\t%d\t%d\n', ...
        results{k,1}, ...
        results{k,2}, ...
        results{k,3}, ...
        results{k,4}, ...
        results{k,5}, ...
        results{k,6});

end

%% ============================================================
% FINAL CHECK
% ============================================================

if failed_count == 0 && success_count == length(files)

    fprintf('\n============================================\n');
    fprintf('ALL SUBJECTS PASSED FEATURE EXTRACTION\n');
    fprintf('============================================\n');

else

    fprintf('\n============================================\n');
    fprintf('WARNING: SOME SUBJECTS FAILED\n');
    fprintf('============================================\n');

end

fprintf('\nOutput folder:\n%s\n',output_folder);