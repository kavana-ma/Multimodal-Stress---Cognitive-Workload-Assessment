%% VALIDATE 42-SECOND N-BACK EXTRACTION - VP001
clear; clc;

base = 'D:\major_project_group50\dataset';

subject = 'VP001';

WINDOW_SEC = 42;

EEG_FS  = 200;
NIRS_FS = 10;

EEG_SAMPLES  = WINDOW_SEC * EEG_FS;   % 8400
NIRS_SAMPLES = WINDOW_SEC * NIRS_FS;  % 420

%% ============================================================
% LOAD EEG
% ============================================================

eeg_data = load(fullfile( ...
    base, [subject '-EEG'], 'cnt_nback.mat'));

cnt_eeg = eeg_data.cnt_nback;

% Keep only 28 EEG channels
eeg = cnt_eeg.x(:,1:28);

%% Load EEG markers
eeg_mrk_data = load(fullfile( ...
    base, [subject '-EEG'], 'mrk_nback.mat'));

mrk_eeg = eeg_mrk_data.mrk_nback;

%% Find session markers
session_codes = [112 128 144];

session_idx = [];

for k = 1:length(mrk_eeg.event.desc)

    if ismember(mrk_eeg.event.desc(k), session_codes)
        session_idx(end+1) = k;
    end

end

%% ============================================================
% LOAD fNIRS
% ============================================================

nirs_data = load(fullfile( ...
    base, [subject '-NIRS'], 'cnt_nback.mat'));

cnt_nirs = nirs_data.cnt_nback;

oxy   = cnt_nirs.oxy.x;
deoxy = cnt_nirs.deoxy.x;

%% Load NIRS markers
nirs_mrk_data = load(fullfile( ...
    base, [subject '-NIRS'], 'mrk_nback.mat'));

mrk_nirs = nirs_mrk_data.mrk_nback;

%% ============================================================
% BASIC CHECK
% ============================================================

fprintf('\n============================================\n');
fprintf(' VP001 EXTRACTION VALIDATION\n');
fprintf('============================================\n\n');

fprintf('EEG signal      : %d samples x %d channels\n', ...
    size(eeg,1), size(eeg,2));

fprintf('fNIRS oxy       : %d samples x %d channels\n', ...
    size(oxy,1), size(oxy,2));

fprintf('fNIRS deoxy     : %d samples x %d channels\n', ...
    size(deoxy,1), size(deoxy,2));

fprintf('\nNumber EEG blocks  : %d\n', length(session_idx));
fprintf('Number NIRS blocks : %d\n', length(mrk_nirs.event.desc));

%% ============================================================
% CHECK ALL 27 BLOCKS
% ============================================================

fprintf('\n');
fprintf('------------------------------------------------------------\n');
fprintf('Block | Class | EEG start | EEG end | NIRS start | NIRS end\n');
fprintf('------------------------------------------------------------\n');

all_valid = true;

for b = 1:27

    %% --------------------------------------------------------
    % EEG START
    % ---------------------------------------------------------

    eeg_marker_raw = ...
        mrk_eeg.time(session_idx(b));

    eeg_start_sec = eeg_marker_raw / 1000;

    eeg_start_sample = ...
        round(eeg_start_sec * EEG_FS) + 1;

    eeg_end_sample = ...
        eeg_start_sample + EEG_SAMPLES - 1;

    %% --------------------------------------------------------
    % fNIRS START
    % ---------------------------------------------------------

    nirs_marker_raw = ...
        mrk_nirs.time(b);

    nirs_start_sec = nirs_marker_raw / 1000;

    nirs_start_sample = ...
        round(nirs_start_sec * NIRS_FS) + 1;

    nirs_end_sample = ...
        nirs_start_sample + NIRS_SAMPLES - 1;

    %% --------------------------------------------------------
    % CLASS
    % ---------------------------------------------------------

    code = mrk_eeg.event.desc(session_idx(b));

    if code == 112
        class_name = '0-back';
    elseif code == 128
        class_name = '2-back';
    elseif code == 144
        class_name = '3-back';
    else
        class_name = 'UNKNOWN';
        all_valid = false;
    end

    %% --------------------------------------------------------
    % BOUNDARY CHECK
    % ---------------------------------------------------------

    eeg_ok = ...
        eeg_start_sample >= 1 && ...
        eeg_end_sample <= size(eeg,1);

    nirs_ok = ...
        nirs_start_sample >= 1 && ...
        nirs_end_sample <= size(oxy,1);

    if ~eeg_ok || ~nirs_ok
        all_valid = false;
    end

    %% --------------------------------------------------------
    % PRINT
    % ---------------------------------------------------------

    fprintf( ...
        '%02d    | %-6s | %8d | %7d | %10d | %8d\n', ...
        b, ...
        class_name, ...
        eeg_start_sample, ...
        eeg_end_sample, ...
        nirs_start_sample, ...
        nirs_end_sample);

end

%% ============================================================
% ACTUALLY EXTRACT SELECTED BLOCKS
% ============================================================

fprintf('\n');
fprintf('============================================\n');
fprintf(' ARRAY SHAPE VALIDATION\n');
fprintf('============================================\n');

test_blocks = [1 2 9 10 18 19 27];

for b = test_blocks

    %% EEG

    eeg_marker_raw = ...
        mrk_eeg.time(session_idx(b));

    eeg_start_sec = eeg_marker_raw / 1000;

    eeg_start_sample = ...
        round(eeg_start_sec * EEG_FS) + 1;

    eeg_end_sample = ...
        eeg_start_sample + EEG_SAMPLES - 1;

    eeg_epoch = ...
        eeg(eeg_start_sample:eeg_end_sample,:)';

    %% fNIRS

    nirs_marker_raw = ...
        mrk_nirs.time(b);

    nirs_start_sec = nirs_marker_raw / 1000;

    nirs_start_sample = ...
        round(nirs_start_sec * NIRS_FS) + 1;

    nirs_end_sample = ...
        nirs_start_sample + NIRS_SAMPLES - 1;

    oxy_epoch = ...
        oxy(nirs_start_sample:nirs_end_sample,:)';

    deoxy_epoch = ...
        deoxy(nirs_start_sample:nirs_end_sample,:)';

    fnirs_epoch = zeros( ...
        36, NIRS_SAMPLES, 2, 'single');

    fnirs_epoch(:,:,1) = single(oxy_epoch);
    fnirs_epoch(:,:,2) = single(deoxy_epoch);

    %% Print shapes

    fprintf('\nBlock %02d\n', b);

    fprintf('  EEG   shape = [%d %d]\n', ...
        size(eeg_epoch,1), ...
        size(eeg_epoch,2));

    fprintf('  fNIRS shape = [%d %d %d]\n', ...
        size(fnirs_epoch,1), ...
        size(fnirs_epoch,2), ...
        size(fnirs_epoch,3));

end

%% ============================================================
% FINAL RESULT
% ============================================================

fprintf('\n============================================\n');

if all_valid
    fprintf('ALL BOUNDARY CHECKS PASSED\n');
else
    fprintf('BOUNDARY CHECK FAILED\n');
end

fprintf('============================================\n');