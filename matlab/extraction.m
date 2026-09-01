%% ============================================================
% EXTRACT SYNCHRONIZED EEG + fNIRS VF/BL EPOCHS
% ALL 24 PAIRED SUBJECTS
% ============================================================

clear;
clc;

basePath = "D:\major_project_group50\dataset";

%% ============================================================
% PARAMETERS
% ============================================================

preTime  = 5;       % seconds before event
postTime = 15;      % seconds after event

fprintf('\n============================================\n');
fprintf('SYNCHRONIZED EPOCH EXTRACTION\n');
fprintf('============================================\n');

fprintf('Epoch window = %.1f to +%.1f sec\n', ...
    -preTime, postTime);

%% ============================================================
% FIND SUBJECT FOLDERS
% ============================================================

allFolders = dir(basePath);

EEG_subjects = {};
NIRS_subjects = {};

for i = 1:length(allFolders)

    if allFolders(i).isdir && ...
       ~strcmp(allFolders(i).name,'.') && ...
       ~strcmp(allFolders(i).name,'..')

        name = allFolders(i).name;

        if endsWith(name,"-EEG")
            EEG_subjects{end+1} = erase(name,"-EEG");
        end

        if endsWith(name,"-NIRS")
            NIRS_subjects{end+1} = erase(name,"-NIRS");
        end
    end
end

%% ============================================================
% PAIRED SUBJECTS
% ============================================================

pairedSubjects = intersect(EEG_subjects,NIRS_subjects);

fprintf('\nPaired subjects = %d\n', ...
    length(pairedSubjects));

disp(pairedSubjects');

%% ============================================================
% OUTPUT DIRECTORY
% ============================================================

outputDir = fullfile(basePath,"synchronized_epochs");

if ~exist(outputDir,'dir')
    mkdir(outputDir);
end

fprintf('\nOutput directory:\n%s\n',outputDir);

%% ============================================================
% PROCESS EACH SUBJECT
% ============================================================

for s = 1:length(pairedSubjects)

    subject = pairedSubjects{s};

    fprintf('\n');
    fprintf('============================================\n');
    fprintf('SUBJECT %s (%d/%d)\n', ...
        subject,s,length(pairedSubjects));
    fprintf('============================================\n');

    %% --------------------------------------------------------
    % FILE PATHS
    % --------------------------------------------------------

    EEG_path = fullfile( ...
        basePath,subject + "-EEG");

    NIRS_path = fullfile( ...
        basePath,subject + "-NIRS");

    eegCntFile = fullfile(EEG_path,"cnt_vf.mat");
    eegMrkFile = fullfile(EEG_path,"mrk_vf.mat");

    nirsCntFile = fullfile(NIRS_path,"cnt_vf.mat");
    nirsMrkFile = fullfile(NIRS_path,"mrk_vf.mat");

    %% --------------------------------------------------------
    % CHECK FILES
    % --------------------------------------------------------

    if ~(isfile(eegCntFile) && ...
         isfile(eegMrkFile) && ...
         isfile(nirsCntFile) && ...
         isfile(nirsMrkFile))

        fprintf('Required file missing. SKIPPING.\n');
        continue;
    end

    %% --------------------------------------------------------
    % LOAD EEG
    % --------------------------------------------------------

    S = load(eegCntFile);
    cnt_EEG = S.cnt_vf;

    S = load(eegMrkFile);
    mrk_EEG = S.mrk_vf;

    %% --------------------------------------------------------
    % LOAD NIRS
    % --------------------------------------------------------

    S = load(nirsCntFile);
    cnt_NIRS = S.cnt_vf;

    S = load(nirsMrkFile);
    mrk_NIRS = S.mrk_vf;

    %% --------------------------------------------------------
    % SAMPLING RATES
    % --------------------------------------------------------

    eeg_fs = cnt_EEG.fs;
    nirs_fs = cnt_NIRS.oxy.fs;

    fprintf('EEG fs  = %.2f Hz\n',eeg_fs);
    fprintf('NIRS fs = %.2f Hz\n',nirs_fs);

    %% --------------------------------------------------------
    % MARKER TIMES
    %
    % Marker values are milliseconds.
    % --------------------------------------------------------

    EEG_time_sec = mrk_EEG.time / 1000;
    NIRS_time_sec = mrk_NIRS.time / 1000;

    %% --------------------------------------------------------
    % CHECK EVENT COUNT
    % --------------------------------------------------------

    nEvents = length(EEG_time_sec);

    if nEvents ~= 60 || length(NIRS_time_sec) ~= 60

        fprintf('Unexpected number of events. SKIPPING.\n');
        continue;
    end

    %% ========================================================
    % CALCULATE SUBJECT-SPECIFIC SEGMENT OFFSETS
    % ========================================================

    delta = NIRS_time_sec - EEG_time_sec;

    offset1 = median(delta(1:20));
    offset2 = median(delta(21:40));
    offset3 = median(delta(41:60));

    fprintf('\nSegment offsets:\n');
    fprintf('Events 1-20  = %.4f sec\n',offset1);
    fprintf('Events 21-40 = %.4f sec\n',offset2);
    fprintf('Events 41-60 = %.4f sec\n',offset3);

    %% ========================================================
    % ALIGN NIRS EVENT TIMES
    % ========================================================

    NIRS_aligned_time = zeros(size(NIRS_time_sec));

    NIRS_aligned_time(1:20) = ...
        NIRS_time_sec(1:20) - offset1;

    NIRS_aligned_time(21:40) = ...
        NIRS_time_sec(21:40) - offset2;

    NIRS_aligned_time(41:60) = ...
        NIRS_time_sec(41:60) - offset3;

    %% ========================================================
    % RESIDUAL CHECK
    % ========================================================

    residual = NIRS_aligned_time - EEG_time_sec;

    fprintf('\nAlignment residual:\n');
    fprintf('Mean = %.6f sec\n',mean(residual));
    fprintf('Std  = %.6f sec\n',std(residual));
    fprintf('Max  = %.6f sec\n',max(abs(residual)));

    %% ========================================================
    % EPOCH SAMPLE COUNTS
    % ========================================================

    eeg_pre_samples = round(preTime * eeg_fs);
    eeg_post_samples = round(postTime * eeg_fs);

    nirs_pre_samples = round(preTime * nirs_fs);
    nirs_post_samples = round(postTime * nirs_fs);

    nEEGSamples = eeg_pre_samples + eeg_post_samples + 1;
    nNIRSSamples = nirs_pre_samples + nirs_post_samples + 1;

    fprintf('\nEpoch sizes:\n');
    fprintf('EEG  = %d samples\n',nEEGSamples);
    fprintf('NIRS = %d samples\n',nNIRSSamples);

    %% ========================================================
    % PREALLOCATE
    % ========================================================

    nEEGChannels = size(cnt_EEG.x,2);

    nNIRSChannels = size(cnt_NIRS.oxy.x,2);

    EEG_epochs = nan( ...
        nEEGSamples, ...
        nEEGChannels, ...
        nEvents);

    Oxy_epochs = nan( ...
        nNIRSSamples, ...
        nNIRSChannels, ...
        nEvents);

    Deoxy_epochs = nan( ...
        nNIRSSamples, ...
        nNIRSChannels, ...
        nEvents);

    %% ========================================================
    % EVENT LABELS
    % ========================================================

    eventClass = strings(nEvents,1);

    %% ========================================================
    % EXTRACT EACH EVENT
    % ========================================================

    validEvent = true(nEvents,1);

    for e = 1:nEvents

        %% ----------------------------------------------------
        % CLASS
        % ----------------------------------------------------

        eeg_class = find(mrk_EEG.y(:,e) ~= 0);

        if isempty(eeg_class)

            validEvent(e) = false;
            continue;

        end

        eventClass(e) = ...
            string(mrk_EEG.className{eeg_class(1)});

        %% ----------------------------------------------------
        % EEG EVENT
        % ----------------------------------------------------

        eegEventSample = ...
            round(EEG_time_sec(e) * eeg_fs) + 1;

        eegStart = ...
            eegEventSample - eeg_pre_samples;

        eegEnd = ...
            eegEventSample + eeg_post_samples;

        %% ----------------------------------------------------
        % NIRS EVENT
        % ----------------------------------------------------

        nirsEventSample = ...
            round(NIRS_aligned_time(e) * nirs_fs) + 1;

        nirsStart = ...
            nirsEventSample - nirs_pre_samples;

        nirsEnd = ...
            nirsEventSample + nirs_post_samples;

        %% ----------------------------------------------------
        % CHECK BOUNDARIES
        % ----------------------------------------------------

        if eegStart < 1 || ...
           eegEnd > size(cnt_EEG.x,1)

            validEvent(e) = false;
            continue;
        end

        if nirsStart < 1 || ...
           nirsEnd > size(cnt_NIRS.oxy.x,1)

            validEvent(e) = false;
            continue;
        end

        %% ----------------------------------------------------
        % EXTRACT EEG
        % ----------------------------------------------------

        EEG_epochs(:,:,e) = ...
            cnt_EEG.x(eegStart:eegEnd,:);

        %% ----------------------------------------------------
        % EXTRACT OXY
        % ----------------------------------------------------

        Oxy_epochs(:,:,e) = ...
            cnt_NIRS.oxy.x(nirsStart:nirsEnd,:);

        %% ----------------------------------------------------
        % EXTRACT DEOXY
        % ----------------------------------------------------

        Deoxy_epochs(:,:,e) = ...
            cnt_NIRS.deoxy.x(nirsStart:nirsEnd,:);

    end

    %% ========================================================
    % REMOVE INVALID EVENTS
    % ========================================================

    EEG_epochs = EEG_epochs(:,:,validEvent);

    Oxy_epochs = Oxy_epochs(:,:,validEvent);

    Deoxy_epochs = Deoxy_epochs(:,:,validEvent);

    eventClass = eventClass(validEvent);

    EEG_event_time = EEG_time_sec(validEvent);

    NIRS_event_time_raw = NIRS_time_sec(validEvent);

    NIRS_event_time_aligned = ...
        NIRS_aligned_time(validEvent);

    residual_valid = residual(validEvent);

    %% ========================================================
    % TIME VECTORS
    % ========================================================

    EEG_time_axis = ...
        (-eeg_pre_samples:eeg_post_samples) / eeg_fs;

    NIRS_time_axis = ...
        (-nirs_pre_samples:nirs_post_samples) / nirs_fs;

    %% ========================================================
    % SUBJECT INFORMATION
    % ========================================================

    SubjectID = subject;

    %% ========================================================
    % SAVE
    % ========================================================

    outputFile = fullfile( ...
        outputDir, ...
        subject + "_synchronized_epochs.mat");

    save(outputFile, ...
        "SubjectID", ...
        "EEG_epochs", ...
        "Oxy_epochs", ...
        "Deoxy_epochs", ...
        "eventClass", ...
        "EEG_event_time", ...
        "NIRS_event_time_raw", ...
        "NIRS_event_time_aligned", ...
        "residual_valid", ...
        "EEG_time_axis", ...
        "NIRS_time_axis", ...
        "eeg_fs", ...
        "nirs_fs", ...
        "offset1", ...
        "offset2", ...
        "offset3", ...
        "-v7.3");

    %% ========================================================
    % REPORT
    % ========================================================

    nVF = sum(eventClass == "VF");
    nBL = sum(eventClass == "BL");

    fprintf('\nExtracted:\n');
    fprintf('Total events = %d\n',length(eventClass));
    fprintf('VF events    = %d\n',nVF);
    fprintf('BL events    = %d\n',nBL);

    fprintf('\nSaved:\n%s\n',outputFile);

end

%% ============================================================
% FINISHED
% ============================================================

fprintf('\n\n============================================\n');
fprintf('EPOCH EXTRACTION COMPLETE\n');
fprintf('============================================\n');

fprintf('Output folder:\n%s\n',outputDir);