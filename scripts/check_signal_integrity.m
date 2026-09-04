%% CHECK EEG + fNIRS SIGNAL INTEGRITY
clear; clc;

base = 'D:\major_project_group50\dataset';

subjects = { ...
    'VP001','VP002','VP003','VP004','VP005','VP006', ...
    'VP007','VP008','VP009','VP010','VP011', ...
    'VP014','VP015','VP016','VP017','VP018','VP019', ...
    'VP020','VP021','VP022','VP023','VP024','VP025','VP026'};

fprintf('\n============================================\n');
fprintf(' SIGNAL INTEGRITY CHECK\n');
fprintf('============================================\n\n');

for s = 1:length(subjects)

    subject = subjects{s};

    %% EEG
    eeg_file = fullfile(base, [subject '-EEG'], 'cnt_nback.mat');
    eeg_data = load(eeg_file);
    cnt_eeg = eeg_data.cnt_nback;

    % Channels 1:28 are EEG.
    % Channels 29:30 are HEOG and VEOG.
    eeg = cnt_eeg.x(:,1:28);

    eeg_nan = sum(isnan(eeg(:)));
    eeg_inf = sum(isinf(eeg(:)));

    %% fNIRS
    nirs_file = fullfile(base, [subject '-NIRS'], 'cnt_nback.mat');
    nirs_data = load(nirs_file);
    cnt_nirs = nirs_data.cnt_nback;

    oxy = cnt_nirs.oxy.x;
    deoxy = cnt_nirs.deoxy.x;

    nirs_nan = sum(isnan(oxy(:))) + sum(isnan(deoxy(:)));
    nirs_inf = sum(isinf(oxy(:))) + sum(isinf(deoxy(:)));

    %% Channel counts
    eeg_channels = size(eeg,2);
    nirs_channels = size(oxy,2);

    fprintf('%s\n', subject);

    fprintf('  EEG  : %d channels | NaN=%d | Inf=%d\n', ...
        eeg_channels, eeg_nan, eeg_inf);

    fprintf('  fNIRS: %d channels | NaN=%d | Inf=%d\n', ...
        nirs_channels, nirs_nan, nirs_inf);

end

fprintf('\n============================================\n');
fprintf('DONE\n');
fprintf('============================================\n');