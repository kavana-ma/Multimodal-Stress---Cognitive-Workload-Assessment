%% 04_bandpass_filter.m

clc;

load('workspace_step3.mat');

% Band-pass filter
lowCut = 0.01;
highCut = 0.20;

[b,a] = butter(4,[lowCut highCut]/(fs/2),'bandpass');

HbO_filtered = cell(size(HbO_epochs));
HbR_filtered = cell(size(HbR_epochs));

for i = 1:length(HbO_epochs)

    HbO_filtered{i} = filtfilt(b,a,HbO_epochs{i});
    HbR_filtered{i} = filtfilt(b,a,HbR_epochs{i});

end

fprintf('\n===== FILTERING COMPLETE =====\n');
fprintf('Filtered Epoch Size : %d x %d\n',size(HbO_filtered{1},1),size(HbO_filtered{1},2));

save('workspace_step4.mat');
