%% 06_create_tensor.m
% Create final tensor for deep learning

clc;

load('workspace_step5.mat');

numEpochs   = length(HbO_norm);
numChannels = size(HbO_norm{1},2);
numSamples  = size(HbO_norm{1},1);

% Tensor: Epoch × Channel × Sample × Signal
X_fnirs = zeros(numEpochs, numChannels, numSamples, 2);

for i = 1:numEpochs

    % Transpose because MATLAB stores Sample × Channel
    X_fnirs(i,:,:,1) = HbO_norm{i}';
    X_fnirs(i,:,:,2) = HbR_norm{i}';

end

y = epochLabels(:);

fprintf('\n===== TENSOR CREATED =====\n');
fprintf('Tensor Shape : [%d %d %d %d]\n', size(X_fnirs));
fprintf('Labels Shape : [%d %d]\n', size(y));

% Save processed data
if ~exist('output','dir')
    mkdir('output')
end

save('output/fnirs_processed.mat',...
    'X_fnirs','y','channels','fs',...
    '-v7.3');

disp('Saved: output/fnirs_processed.mat')