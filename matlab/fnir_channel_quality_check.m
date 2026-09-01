%% ============================================================
% F NIRS CHANNEL QUALITY CHECK - ALL 24 SUBJECTS
% ============================================================

clear;
clc;

inputFolder = ...
    "D:\major_project_group50\dataset\synchronized_epochs";

subjects = { ...
    'VP001','VP002','VP003','VP004','VP005','VP006',...
    'VP007','VP008','VP009','VP010','VP011','VP014',...
    'VP015','VP016','VP017','VP018','VP019','VP020',...
    'VP021','VP022','VP023','VP024','VP025','VP026'};

fprintf('\n============================================\n');
fprintf('fNIRS CHANNEL QUALITY CHECK\n');
fprintf('============================================\n');

for s = 1:length(subjects)

    subject = subjects{s};

    file = fullfile(inputFolder, ...
        subject + "_synchronized_epochs.mat");

    load(file);

    fprintf('\n--------------------------------------------\n');
    fprintf('%s\n',subject);
    fprintf('--------------------------------------------\n');

    %% OXY

    nChannels = size(Oxy_epochs,2);

    fprintf('Oxy channels = %d\n',nChannels);

    fprintf('\nOxy channel statistics:\n');

    fprintf('Channel\tMean\t\tStd\t\tMin\t\tMax\n');

    for ch = 1:nChannels

        data = Oxy_epochs(:,ch,:);

        fprintf('%d\t%.6f\t%.6f\t%.6f\t%.6f\n', ...
            ch, ...
            mean(data(:)), ...
            std(data(:)), ...
            min(data(:)), ...
            max(data(:)));

    end

    %% DEOXY

    nChannels = size(Deoxy_epochs,2);

    fprintf('\nDeoxy channels = %d\n',nChannels);

    fprintf('\nDeoxy channel statistics:\n');

    fprintf('Channel\tMean\t\tStd\t\tMin\t\tMax\n');

    for ch = 1:nChannels

        data = Deoxy_epochs(:,ch,:);

        fprintf('%d\t%.6f\t%.6f\t%.6f\t%.6f\n', ...
            ch, ...
            mean(data(:)), ...
            std(data(:)), ...
            min(data(:)), ...
            max(data(:)));

    end

end

fprintf('\n============================================\n');
fprintf('CHANNEL QUALITY CHECK COMPLETE\n');
fprintf('============================================\n');