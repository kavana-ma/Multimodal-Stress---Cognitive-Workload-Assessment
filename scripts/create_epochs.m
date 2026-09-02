%% 03_create_epochs.m

clc;

load('workspace_step2.mat');

epochLength = 30*fs;      % 30 s = 300 samples

HbO_epochs = {};
HbR_epochs = {};
epochLabels = [];

for i = 1:length(markerSamples)

    startIdx = markerSamples(i);
    endIdx = startIdx + epochLength - 1;

    if endIdx > size(HbO,1)
        continue
    end

    HbO_epochs{end+1} = HbO(startIdx:endIdx,:);
    HbR_epochs{end+1} = HbR(startIdx:endIdx,:);
    epochLabels(end+1) = labels(i);

end

fprintf('\n===== EPOCHING COMPLETE =====\n');
fprintf('Total Epochs : %d\n',length(HbO_epochs));
fprintf('Epoch Size   : %d x %d\n',size(HbO_epochs{1},1),size(HbO_epochs{1},2));

save('workspace_step3.mat');