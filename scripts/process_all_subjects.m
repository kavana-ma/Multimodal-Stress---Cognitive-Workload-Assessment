%% 07_process_all_subjects.m
% Process all TU Berlin fNIRS subjects

clc; clear;

rootFolder = 'C:\Users\Natar\OneDrive\Desktop\Major project\data';

subjects = dir(fullfile(rootFolder,'VP*-NIRS'));

X_all = [];
y_all = [];
subjectID = [];
trialID = [];

for s = 1:length(subjects)

    folder = fullfile(rootFolder,subjects(s).name);

    % Load files
    cnt = load(fullfile(folder,'cnt_nback.mat'));
    mrk = load(fullfile(folder,'mrk_nback.mat'));

    cntData = cnt.cnt_nback;
    mrkData = mrk.mrk_nback;

    % Signals
    HbO = cntData.oxy.x;
    HbR = cntData.deoxy.x;

    fs = cntData.oxy.fs;

    % Labels
    markerTime = mrkData.time(:);
    [~,labels] = max(mrkData.y,[],1);
    labels = labels(:)-1;

    markerSamples = round((markerTime/1000)*fs);

    epochLength = 30*fs;

    % Filter
    [b,a] = butter(4,[0.01 0.20]/(fs/2),'bandpass');

    for e = 1:length(markerSamples)

        st = markerSamples(e);
        ed = st + epochLength - 1;

        if ed > size(HbO,1)
            continue;
        end

        oxy = HbO(st:ed,:);
        deoxy = HbR(st:ed,:);

        % Band-pass
        oxy = filtfilt(b,a,oxy);
        deoxy = filtfilt(b,a,deoxy);

        % Baseline normalization
        oxy = oxy - mean(oxy(1:50,:),1);
        deoxy = deoxy - mean(deoxy(1:50,:),1);

        sample = zeros(36,300,2);
        sample(:,:,1) = oxy';
        sample(:,:,2) = deoxy';

        X_all = cat(1,X_all,reshape(sample,1,36,300,2));

        y_all(end+1,1) = labels(e);
        subjectID(end+1,1) = s;
        trialID(end+1,1) = e;

    end

    fprintf('%s completed\n',subjects(s).name);

end

X_fnirs = X_all;
y = y_all;

if ~exist('output','dir')
    mkdir('output')
end

save('output/fnirs_all_subjects.mat',...
    'X_fnirs','y','subjectID','trialID','-v7.3');

disp(size(X_fnirs))
disp(size(y))