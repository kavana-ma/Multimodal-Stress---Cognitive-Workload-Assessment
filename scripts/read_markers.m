%% 02_read_markers.m

clc;

load('workspace_step1.mat');

% Marker times (milliseconds)
markerTime = mrkData.time(:);

% Convert one-hot labels
[~,labels] = max(mrkData.y,[],1);
labels = labels(:)-1;

% Convert ms → samples
markerSamples = round((markerTime/1000)*fs);

markerTable = table(markerSamples,labels,...
    'VariableNames',{'StartSample','Label'});

disp(markerTable)

save('workspace_step2.mat');