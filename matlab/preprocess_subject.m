function preprocess_subject(subjectID, inputFolder, outputFolder)

%% ============================================================
% PREPROCESS SUBJECT
%
% Generalized preprocessing pipeline for synchronized EEG + fNIRS
%
% Input:
%   <inputFolder>/<subjectID>_synchronized_epochs.mat
%
% Output:
%   <outputFolder>/<subjectID>_preprocessed.mat
%
% EEG:
%   Original channels = 30
%   Retained channels = 28
%   HEOG and VEOG removed
%   Band-pass = 1-45 Hz
%   Baseline = -5 to 0 sec
%
% fNIRS:
%   Oxy = 36 channels
%   Deoxy = 36 channels
%   Band-pass = 0.01-0.1 Hz
%   Baseline = -5 to 0 sec
%
% IMPORTANT:
%   This implementation does NOT require Signal Processing Toolbox.
%
% ============================================================


%% ============================================================
% DISPLAY HEADER
% ============================================================

fprintf('\n');
fprintf('============================================\n');
fprintf('%s PREPROCESSING\n', subjectID);
fprintf('============================================\n');


%% ============================================================
% INPUT / OUTPUT FILES
% ============================================================

inputFile = fullfile( ...
    inputFolder, ...
    subjectID + "_synchronized_epochs.mat");

outputFile = fullfile( ...
    outputFolder, ...
    subjectID + "_preprocessed.mat");


%% ============================================================
% CHECK INPUT FILE
% ============================================================

if ~isfile(inputFile)

    error( ...
        'Input synchronized file does not exist:\n%s', ...
        inputFile);

end


%% ============================================================
% CREATE OUTPUT FOLDER
% ============================================================

if ~isfolder(outputFolder)

    mkdir(outputFolder);

end


%% ============================================================
% LOAD SYNCHRONIZED DATA
% ============================================================

load(inputFile);


fprintf('\nLoaded subject: %s\n', subjectID);


%% ============================================================
% CHECK REQUIRED VARIABLES
% ============================================================

requiredVariables = { ...
    'EEG_epochs', ...
    'Oxy_epochs', ...
    'Deoxy_epochs', ...
    'EEG_time_axis', ...
    'NIRS_time_axis', ...
    'eventClass', ...
    'eeg_fs', ...
    'nirs_fs' ...
    };

for k = 1:length(requiredVariables)

    if ~exist(requiredVariables{k}, 'var')

        error( ...
            'Required variable "%s" is missing from %s', ...
            requiredVariables{k}, ...
            inputFile);

    end

end


%% ============================================================
% INPUT DATA INFORMATION
% ============================================================

fprintf('\n========== INPUT DATA ==========\n');

fprintf('EEG size    : %d x %d x %d\n', ...
    size(EEG_epochs,1), ...
    size(EEG_epochs,2), ...
    size(EEG_epochs,3));

fprintf('Oxy size    : %d x %d x %d\n', ...
    size(Oxy_epochs,1), ...
    size(Oxy_epochs,2), ...
    size(Oxy_epochs,3));

fprintf('Deoxy size  : %d x %d x %d\n', ...
    size(Deoxy_epochs,1), ...
    size(Deoxy_epochs,2), ...
    size(Deoxy_epochs,3));

fprintf('EEG fs      : %.2f Hz\n', eeg_fs);
fprintf('NIRS fs     : %.2f Hz\n', nirs_fs);

nEvents = size(EEG_epochs,3);

fprintf('Number events = %d\n', nEvents);


%% ============================================================
% CHECK DATA CONSISTENCY
% ============================================================

if size(Oxy_epochs,3) ~= nEvents

    error('EEG and Oxy event counts do not match.');

end

if size(Deoxy_epochs,3) ~= nEvents

    error('EEG and Deoxy event counts do not match.');

end

if length(eventClass) ~= nEvents

    error('eventClass count does not match number of events.');

end


%% ============================================================
% ============================================================
% EEG PREPROCESSING
% ============================================================
%% ============================================================

fprintf('\n');
fprintf('============================================\n');
fprintf('EEG PREPROCESSING\n');
fprintf('============================================\n');


%% ------------------------------------------------------------
% RETAIN EEG CHANNELS
% ------------------------------------------------------------

% Original EEG:
%
% 1-28 = EEG channels
% 29   = HEOG
% 30   = VEOG
%
% Therefore retain channels 1:28.

if size(EEG_epochs,2) < 28

    error('EEG data contains fewer than 28 channels.');

end


EEG_data = EEG_epochs(:,1:28,:);

fprintf('EEG channels retained = %d\n', ...
    size(EEG_data,2));


%% ------------------------------------------------------------
% EEG BAND-PASS
% ------------------------------------------------------------

EEG_low_cutoff  = 1;
EEG_high_cutoff = 45;

fprintf('\nEEG band-pass = %.0f-%.0f Hz\n', ...
    EEG_low_cutoff, EEG_high_cutoff);


EEG_filtered = fft_bandpass_3d( ...
    EEG_data, ...
    eeg_fs, ...
    EEG_low_cutoff, ...
    EEG_high_cutoff);


%% ------------------------------------------------------------
% EEG BASELINE
% ------------------------------------------------------------

baseline = EEG_time_axis >= -5 & EEG_time_axis <= 0;

baseline_idx_EEG = baseline;

if sum(baseline_idx_EEG) == 0

    error('EEG baseline interval -5 to 0 sec not found.');

end


EEG_baseline_samples = sum(baseline_idx_EEG);

fprintf('EEG baseline samples = %d\n', ...
    EEG_baseline_samples);


%% ------------------------------------------------------------
% EEG BASELINE CORRECTION
% ------------------------------------------------------------

EEG_baseline_check = ...
    mean( ...
        EEG_filtered(baseline_idx_EEG,:,:), ...
        1);


EEG_preprocessed = ...
    EEG_filtered - EEG_baseline_check;


fprintf('EEG baseline correction completed.\n');


%% ============================================================
% ============================================================
% fNIRS PREPROCESSING
% ============================================================
%% ============================================================

fprintf('\n');
fprintf('============================================\n');
fprintf('fNIRS PREPROCESSING\n');
fprintf('============================================\n');


%% ------------------------------------------------------------
% fNIRS BAND-PASS
% ------------------------------------------------------------

NIRS_low_cutoff  = 0.01;
NIRS_high_cutoff = 0.1;

fprintf('\nfNIRS band-pass = %.2f-%.2f Hz\n', ...
    NIRS_low_cutoff, NIRS_high_cutoff);


%% ------------------------------------------------------------
% OXYGENATED HbO
% ------------------------------------------------------------

Oxy_filtered = fft_bandpass_3d( ...
    Oxy_epochs, ...
    nirs_fs, ...
    NIRS_low_cutoff, ...
    NIRS_high_cutoff);


%% ------------------------------------------------------------
% DEOXYGENATED HbR
% ------------------------------------------------------------

Deoxy_filtered = fft_bandpass_3d( ...
    Deoxy_epochs, ...
    nirs_fs, ...
    NIRS_low_cutoff, ...
    NIRS_high_cutoff);


%% ------------------------------------------------------------
% NIRS BASELINE
% ------------------------------------------------------------

baseline_idx_NIRS = ...
    NIRS_time_axis >= -5 & NIRS_time_axis <= 0;


if sum(baseline_idx_NIRS) == 0

    error('NIRS baseline interval -5 to 0 sec not found.');

end


NIRS_baseline_samples = sum(baseline_idx_NIRS);

fprintf('NIRS baseline samples = %d\n', ...
    NIRS_baseline_samples);


%% ------------------------------------------------------------
% OXY BASELINE
% ------------------------------------------------------------

Oxy_baseline_check = ...
    mean( ...
        Oxy_filtered(baseline_idx_NIRS,:,:), ...
        1);


Oxy_preprocessed = ...
    Oxy_filtered - Oxy_baseline_check;


%% ------------------------------------------------------------
% DEOXY BASELINE
% ------------------------------------------------------------

Deoxy_baseline_check = ...
    mean( ...
        Deoxy_filtered(baseline_idx_NIRS,:,:), ...
        1);


Deoxy_preprocessed = ...
    Deoxy_filtered - Deoxy_baseline_check;


fprintf('fNIRS baseline correction completed.\n');


%% ============================================================
% ============================================================
% CLASS INFORMATION
% ============================================================
%% ============================================================

bl_idx = find(eventClass == "BL");
vf_idx = find(eventClass == "VF");


fprintf('\n');
fprintf('============================================\n');
fprintf('EVENT CLASSES\n');
fprintf('============================================\n');

fprintf('BL trials = %d\n', length(bl_idx));
fprintf('VF trials = %d\n', length(vf_idx));


%% ============================================================
% ============================================================
% DATA QUALITY CHECK
% ============================================================
%% ============================================================

fprintf('\n');
fprintf('============================================\n');
fprintf('PREPROCESSED DATA QUALITY\n');
fprintf('============================================\n');


%% ------------------------------------------------------------
% EEG
% ------------------------------------------------------------

EEG_nan = sum(isnan(EEG_preprocessed(:)));
EEG_inf = sum(isinf(EEG_preprocessed(:)));

fprintf('\nEEG:\n');

fprintf('NaN = %d\n', EEG_nan);
fprintf('Inf = %d\n', EEG_inf);

fprintf('Range = %.4f to %.4f\n', ...
    min(EEG_preprocessed(:)), ...
    max(EEG_preprocessed(:)));


%% ------------------------------------------------------------
% OXY
% ------------------------------------------------------------

Oxy_nan = sum(isnan(Oxy_preprocessed(:)));
Oxy_inf = sum(isinf(Oxy_preprocessed(:)));

fprintf('\nOxy:\n');

fprintf('NaN = %d\n', Oxy_nan);
fprintf('Inf = %d\n', Oxy_inf);

fprintf('Range = %.6f to %.6f\n', ...
    min(Oxy_preprocessed(:)), ...
    max(Oxy_preprocessed(:)));


%% ------------------------------------------------------------
% DEOXY
% ------------------------------------------------------------

Deoxy_nan = sum(isnan(Deoxy_preprocessed(:)));
Deoxy_inf = sum(isinf(Deoxy_preprocessed(:)));

fprintf('\nDeoxy:\n');

fprintf('NaN = %d\n', Deoxy_nan);
fprintf('Inf = %d\n', Deoxy_inf);

fprintf('Range = %.6f to %.6f\n', ...
    min(Deoxy_preprocessed(:)), ...
    max(Deoxy_preprocessed(:)));


%% ============================================================
% STOP IF INVALID DATA
% ============================================================

if EEG_nan > 0 || EEG_inf > 0

    error('Invalid EEG values detected.');

end

if Oxy_nan > 0 || Oxy_inf > 0

    error('Invalid Oxy values detected.');

end

if Deoxy_nan > 0 || Deoxy_inf > 0

    error('Invalid Deoxy values detected.');

end


%% ============================================================
% ============================================================
% BASELINE VALIDATION
% ============================================================
%% ============================================================

fprintf('\n');
fprintf('============================================\n');
fprintf('BASELINE CHECK\n');
fprintf('============================================\n');


%% ------------------------------------------------------------
% EEG BASELINE CHECK
% ------------------------------------------------------------

EEG_baseline_after = ...
    mean( ...
        EEG_preprocessed(baseline_idx_EEG,:,:), ...
        1);


mean_abs_EEG_baseline = ...
    mean(abs(EEG_baseline_after(:)));


fprintf('Mean absolute EEG baseline = %.10f\n', ...
    mean_abs_EEG_baseline);


%% ------------------------------------------------------------
% OXY BASELINE CHECK
% ------------------------------------------------------------

Oxy_baseline_after = ...
    mean( ...
        Oxy_preprocessed(baseline_idx_NIRS,:,:), ...
        1);


mean_abs_Oxy_baseline = ...
    mean(abs(Oxy_baseline_after(:)));


fprintf('Mean absolute Oxy baseline = %.10f\n', ...
    mean_abs_Oxy_baseline);


%% ------------------------------------------------------------
% DEOXY BASELINE CHECK
% ------------------------------------------------------------

Deoxy_baseline_after = ...
    mean( ...
        Deoxy_preprocessed(baseline_idx_NIRS,:,:), ...
        1);


mean_abs_Deoxy_baseline = ...
    mean(abs(Deoxy_baseline_after(:)));


fprintf('Mean absolute Deoxy baseline = %.10f\n', ...
    mean_abs_Deoxy_baseline);


%% ============================================================
% ============================================================
% POST-STIMULUS SUMMARY
% ============================================================
%% ============================================================

post_idx_EEG = EEG_time_axis > 0;

post_idx_NIRS = NIRS_time_axis > 0;


%% ------------------------------------------------------------
% EEG SUMMARY
% ------------------------------------------------------------

if ~isempty(bl_idx)

    BL_EEG = EEG_preprocessed( ...
        post_idx_EEG,:,bl_idx);

    BL_EEG_mean = mean(BL_EEG(:));
    BL_EEG_std  = std(BL_EEG(:));

else

    BL_EEG_mean = NaN;
    BL_EEG_std  = NaN;

end


if ~isempty(vf_idx)

    VF_EEG = EEG_preprocessed( ...
        post_idx_EEG,:,vf_idx);

    VF_EEG_mean = mean(VF_EEG(:));
    VF_EEG_std  = std(VF_EEG(:));

else

    VF_EEG_mean = NaN;
    VF_EEG_std = NaN;

end


%% ------------------------------------------------------------
% OXY SUMMARY
% ------------------------------------------------------------

if ~isempty(bl_idx)

    BL_Oxy = Oxy_preprocessed( ...
        post_idx_NIRS,:,bl_idx);

    BL_Oxy_mean = mean(BL_Oxy(:));
    BL_Oxy_std  = std(BL_Oxy(:));

else

    BL_Oxy_mean = NaN;
    BL_Oxy_std  = NaN;

end


if ~isempty(vf_idx)

    VF_Oxy = Oxy_preprocessed( ...
        post_idx_NIRS,:,vf_idx);

    VF_Oxy_mean = mean(VF_Oxy(:));
    VF_Oxy_std  = std(VF_Oxy(:));

else

    VF_Oxy_mean = NaN;
    VF_Oxy_std = NaN;

end


%% ------------------------------------------------------------
% DEOXY SUMMARY
% ------------------------------------------------------------

if ~isempty(bl_idx)

    BL_Deoxy = Deoxy_preprocessed( ...
        post_idx_NIRS,:,bl_idx);

    BL_Deoxy_mean = mean(BL_Deoxy(:));
    BL_Deoxy_std  = std(BL_Deoxy(:));

else

    BL_Deoxy_mean = NaN;
    BL_Deoxy_std = NaN;

end


if ~isempty(vf_idx)

    VF_Deoxy = Deoxy_preprocessed( ...
        post_idx_NIRS,:,vf_idx);

    VF_Deoxy_mean = mean(VF_Deoxy(:));
    VF_Deoxy_std  = std(VF_Deoxy(:));

else

    VF_Deoxy_mean = NaN;
    VF_Deoxy_std = NaN;

end


%% ============================================================
% DISPLAY POST-STIMULUS SUMMARY
% ============================================================

fprintf('\n');
fprintf('============================================\n');
fprintf('POST-STIMULUS SUMMARY\n');
fprintf('============================================\n');

fprintf('\nEEG:\n');

fprintf('BL grand mean = %.4f uV\n', ...
    BL_EEG_mean);

fprintf('VF grand mean = %.4f uV\n', ...
    VF_EEG_mean);

fprintf('BL grand std  = %.4f uV\n', ...
    BL_EEG_std);

fprintf('VF grand std  = %.4f uV\n', ...
    VF_EEG_std);


fprintf('\nOXY:\n');

fprintf('BL grand mean = %.8f\n', ...
    BL_Oxy_mean);

fprintf('VF grand mean = %.8f\n', ...
    VF_Oxy_mean);

fprintf('BL grand std  = %.8f\n', ...
    BL_Oxy_std);

fprintf('VF grand std  = %.8f\n', ...
    VF_Oxy_std);


fprintf('\nDEOXY:\n');

fprintf('BL grand mean = %.8f\n', ...
    BL_Deoxy_mean);

fprintf('VF grand mean = %.8f\n', ...
    VF_Deoxy_mean);

fprintf('BL grand std  = %.8f\n', ...
    BL_Deoxy_std);

fprintf('VF grand std  = %.8f\n', ...
    VF_Deoxy_std);


%% ============================================================
% ============================================================
% FINAL DATA QUALITY
% ============================================================
%% ============================================================

fprintf('\n');
fprintf('============================================\n');
fprintf('FINAL DATA QUALITY\n');
fprintf('============================================\n');

fprintf('EEG NaN   = %d\n', ...
    sum(isnan(EEG_preprocessed(:))));

fprintf('EEG Inf   = %d\n', ...
    sum(isinf(EEG_preprocessed(:))));

fprintf('Oxy NaN   = %d\n', ...
    sum(isnan(Oxy_preprocessed(:))));

fprintf('Oxy Inf   = %d\n', ...
    sum(isinf(Oxy_preprocessed(:))));

fprintf('Deoxy NaN = %d\n', ...
    sum(isnan(Deoxy_preprocessed(:))));

fprintf('Deoxy Inf = %d\n', ...
    sum(isinf(Deoxy_preprocessed(:))));


%% ============================================================
% ============================================================
% PREPARE OUTPUT VARIABLES
% ============================================================
%% ============================================================

SubjectID = char(subjectID);

file = string(inputFile);


%% ------------------------------------------------------------
% Keep synchronized timing variables if available
% ------------------------------------------------------------

if ~exist('EEG_event_time','var')

    EEG_event_time = [];

end


if ~exist('NIRS_event_time_raw','var')

    NIRS_event_time_raw = [];

end


if ~exist('NIRS_event_time_aligned','var')

    NIRS_event_time_aligned = [];

end


if ~exist('offset1','var')

    offset1 = NaN;

end


if ~exist('offset2','var')

    offset2 = NaN;

end


if ~exist('offset3','var')

    offset3 = NaN;

end


if ~exist('residual_valid','var')

    residual_valid = [];

end


%% ============================================================
% ============================================================
% SAVE PREPROCESSED DATA
% ============================================================
%% ============================================================

fprintf('\n');
fprintf('Saving preprocessed data...\n');


save( ...
    outputFile, ...
    'SubjectID', ...
    'EEG_preprocessed', ...
    'Oxy_preprocessed', ...
    'Deoxy_preprocessed', ...
    'EEG_filtered', ...
    'Oxy_filtered', ...
    'Deoxy_filtered', ...
    'EEG_data', ...
    'EEG_time_axis', ...
    'NIRS_time_axis', ...
    'EEG_event_time', ...
    'NIRS_event_time_raw', ...
    'NIRS_event_time_aligned', ...
    'eventClass', ...
    'eeg_fs', ...
    'nirs_fs', ...
    'EEG_baseline_check', ...
    'Oxy_baseline_check', ...
    'Deoxy_baseline_check', ...
    'baseline_idx_EEG', ...
    'baseline_idx_NIRS', ...
    'bl_idx', ...
    'vf_idx', ...
    'offset1', ...
    'offset2', ...
    'offset3', ...
    'residual_valid', ...
    'file', ...
    '-v7.3');


%% ============================================================
% VERIFY OUTPUT
% ============================================================

if ~isfile(outputFile)

    error( ...
        'Output file was not created:\n%s', ...
        outputFile);

end


%% ============================================================
% FINAL MESSAGE
% ============================================================

fprintf('\n');
fprintf('============================================\n');
fprintf('%s PREPROCESSING COMPLETE\n', subjectID);
fprintf('============================================\n');

fprintf('EEG   : %d x %d x %d\n', ...
    size(EEG_preprocessed,1), ...
    size(EEG_preprocessed,2), ...
    size(EEG_preprocessed,3));

fprintf('Oxy   : %d x %d x %d\n', ...
    size(Oxy_preprocessed,1), ...
    size(Oxy_preprocessed,2), ...
    size(Oxy_preprocessed,3));

fprintf('Deoxy : %d x %d x %d\n', ...
    size(Deoxy_preprocessed,1), ...
    size(Deoxy_preprocessed,2), ...
    size(Deoxy_preprocessed,3));

fprintf('\nBL trials = %d\n', length(bl_idx));
fprintf('VF trials = %d\n', length(vf_idx));

fprintf('\nSaved:\n%s\n', outputFile);

fprintf('\n');


end


%% ################################################################
% #################################################################
% LOCAL FUNCTION
% FFT BAND-PASS FILTER
% #################################################################
% #################################################################

function Y = fft_bandpass_3d(X, fs, lowCut, highCut)

%% ============================================================
% FFT BAND-PASS FILTER
%
% This function does not require Signal Processing Toolbox.
%
% Input:
%   X       = samples x channels x trials
%   fs      = sampling frequency
%   lowCut  = lower cutoff
%   highCut = upper cutoff
%
% Output:
%   Y       = band-pass filtered signal
%
% ============================================================


%% ------------------------------------------------------------
% INPUT SIZE
% ------------------------------------------------------------

[nSamples, nChannels, nTrials] = size(X);


%% ------------------------------------------------------------
% FREQUENCY AXIS
% ------------------------------------------------------------

freq = (0:nSamples-1)' * fs / nSamples;


%% ------------------------------------------------------------
% FFT
% ------------------------------------------------------------

Xfft = fft(X,[],1);


%% ------------------------------------------------------------
% CREATE BAND-PASS MASK
% ------------------------------------------------------------

% Positive frequencies
positiveFreq = freq <= fs/2;

% Corresponding negative frequencies
negativeFreq = freq >= fs - fs/2;


passband = ...
    ( ...
        (freq >= lowCut & freq <= highCut) ...
        | ...
        (freq >= fs-highCut & freq <= fs-lowCut) ...
    );


% Make sure DC is removed.
passband(1) = false;


%% ------------------------------------------------------------
% APPLY FILTER
% ------------------------------------------------------------

mask = reshape(passband,[],1,1);

Xfft = Xfft .* mask;


%% ------------------------------------------------------------
% INVERSE FFT
% ------------------------------------------------------------

Y = real(ifft(Xfft,[],1));


%% ------------------------------------------------------------
% PRESERVE ORIGINAL SIZE
% ------------------------------------------------------------

Y = reshape(Y, ...
    nSamples, ...
    nChannels, ...
    nTrials);


end