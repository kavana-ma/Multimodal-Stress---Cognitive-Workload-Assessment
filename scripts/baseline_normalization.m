%% 05_baseline_normalization.m
% Normalize each epoch using first 5-second baseline

clc;

load('workspace_step4.mat');

baselineSamples = 5 * fs;      % 50 samples

HbO_norm = cell(size(HbO_filtered));
HbR_norm = cell(size(HbR_filtered));

for i = 1:length(HbO_filtered)

    oxy = HbO_filtered{i};
    deoxy = HbR_filtered{i};

    % Mean of first 5 seconds for every channel
    oxyBaseline = mean(oxy(1:baselineSamples,:),1);
    deoxyBaseline = mean(deoxy(1:baselineSamples,:),1);

    % Baseline correction
    HbO_norm{i} = oxy - oxyBaseline;
    HbR_norm{i} = deoxy - deoxyBaseline;

end

fprintf('\n===== BASELINE NORMALIZATION COMPLETE =====\n');
fprintf('Normalized Epoch Size : %d x %d\n', ...
    size(HbO_norm{1},1), size(HbO_norm{1},2));

save('workspace_step5.mat');