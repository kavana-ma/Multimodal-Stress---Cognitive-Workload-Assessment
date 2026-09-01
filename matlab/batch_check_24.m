%% ============================================================
% BATCH VERIFY VF FILES AND MARKERS FOR ALL PAIRED SUBJECTS
% ============================================================

clear;
clc;

basePath = "D:\major_project_group50\dataset";

%% ------------------------------------------------------------
% FIND EEG AND NIRS FOLDERS
% ------------------------------------------------------------

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

%% ------------------------------------------------------------
% FIND PAIRED SUBJECTS
% ------------------------------------------------------------

pairedSubjects = intersect(EEG_subjects,NIRS_subjects);

fprintf('\n============================================\n');
fprintf('PAIRED SUBJECTS = %d\n',length(pairedSubjects));
fprintf('============================================\n');

disp(pairedSubjects');

%% ------------------------------------------------------------
% RESULT TABLE
% ------------------------------------------------------------

nSubjects = length(pairedSubjects);

Subject = strings(nSubjects,1);
EEG_cnt = false(nSubjects,1);
EEG_mrk = false(nSubjects,1);
NIRS_cnt = false(nSubjects,1);
NIRS_mrk = false(nSubjects,1);

EEG_events = zeros(nSubjects,1);
NIRS_events = zeros(nSubjects,1);

ClassMatch = false(nSubjects,1);

Offset1 = nan(nSubjects,1);
Offset2 = nan(nSubjects,1);
Offset3 = nan(nSubjects,1);

ResidualMean = nan(nSubjects,1);
ResidualStd = nan(nSubjects,1);
ResidualMax = nan(nSubjects,1);

%% ------------------------------------------------------------
% PROCESS EACH SUBJECT
% ------------------------------------------------------------

for s = 1:nSubjects

    subject = pairedSubjects{s};

    Subject(s) = subject;

    fprintf('\n--------------------------------------------\n');
    fprintf('SUBJECT %s\n',subject);
    fprintf('--------------------------------------------\n');

    EEG_path = fullfile(basePath,subject + "-EEG");
    NIRS_path = fullfile(basePath,subject + "-NIRS");

    %% ========================================================
    % CHECK FILES
    % ========================================================

    eegCntFile = fullfile(EEG_path,"cnt_vf.mat");
    eegMrkFile = fullfile(EEG_path,"mrk_vf.mat");

    nirsCntFile = fullfile(NIRS_path,"cnt_vf.mat");
    nirsMrkFile = fullfile(NIRS_path,"mrk_vf.mat");

    EEG_cnt(s) = isfile(eegCntFile);
    EEG_mrk(s) = isfile(eegMrkFile);

    NIRS_cnt(s) = isfile(nirsCntFile);
    NIRS_mrk(s) = isfile(nirsMrkFile);

    fprintf('EEG cnt_vf.mat  : %d\n',EEG_cnt(s));
    fprintf('EEG mrk_vf.mat  : %d\n',EEG_mrk(s));
    fprintf('NIRS cnt_vf.mat : %d\n',NIRS_cnt(s));
    fprintf('NIRS mrk_vf.mat : %d\n',NIRS_mrk(s));

    %% ========================================================
    % SKIP IF FILE MISSING
    % ========================================================

    if ~(EEG_cnt(s) && EEG_mrk(s) && ...
         NIRS_cnt(s) && NIRS_mrk(s))

        fprintf('Skipping subject because file is missing.\n');
        continue;
    end

    %% ========================================================
    % LOAD EEG
    % ========================================================

    S = load(eegCntFile);
    cnt_EEG = S.cnt_vf;

    S = load(eegMrkFile);
    mrk_EEG = S.mrk_vf;

    %% ========================================================
    % LOAD NIRS
    % ========================================================

    S = load(nirsCntFile);
    cnt_NIRS = S.cnt_vf;

    S = load(nirsMrkFile);
    mrk_NIRS = S.mrk_vf;

    %% ========================================================
    % CHECK EVENT COUNTS
    % ========================================================

    EEG_events(s) = length(mrk_EEG.time);
    NIRS_events(s) = length(mrk_NIRS.time);

    fprintf('EEG events  = %d\n',EEG_events(s));
    fprintf('NIRS events = %d\n',NIRS_events(s));

    %% ========================================================
    % CHECK CLASS AGREEMENT
    % ========================================================

    if EEG_events(s) == NIRS_events(s)

        match = true;

        for e = 1:EEG_events(s)

            eegClass = find(mrk_EEG.y(:,e) ~= 0);
            nirsClass = find(mrk_NIRS.y(:,e) ~= 0);

            if isempty(eegClass) || isempty(nirsClass)

                match = false;
                break;

            end

            if ~strcmp( ...
                    mrk_EEG.className{eegClass(1)}, ...
                    mrk_NIRS.className{nirsClass(1)})

                match = false;
                break;

            end
        end

        ClassMatch(s) = match;

    end

    fprintf('Class agreement = %d\n',ClassMatch(s));

    %% ========================================================
    % TIMESTAMP CONVERSION
    % ========================================================
    %
    % Based on VP001 validation:
    % marker times are milliseconds
    %
    % Therefore:
    % seconds = milliseconds / 1000
    %

    EEG_time_sec = mrk_EEG.time / 1000;
    NIRS_time_sec = mrk_NIRS.time / 1000;

    %% ========================================================
    % ONLY ALIGN IF 60 EVENTS EXIST
    % ========================================================

    if EEG_events(s) == 60 && ...
       NIRS_events(s) == 60 && ...
       ClassMatch(s)

        delta = NIRS_time_sec - EEG_time_sec;

        %% ----------------------------------------------------
        % THREE SEGMENTS
        % ----------------------------------------------------

        Offset1(s) = median(delta(1:20));
        Offset2(s) = median(delta(21:40));
        Offset3(s) = median(delta(41:60));

        %% ----------------------------------------------------
        % APPLY SEGMENT OFFSETS
        % ----------------------------------------------------

        NIRS_aligned = zeros(size(NIRS_time_sec));

        NIRS_aligned(1:20) = ...
            NIRS_time_sec(1:20) - Offset1(s);

        NIRS_aligned(21:40) = ...
            NIRS_time_sec(21:40) - Offset2(s);

        NIRS_aligned(41:60) = ...
            NIRS_time_sec(41:60) - Offset3(s);

        %% ----------------------------------------------------
        % RESIDUAL
        % ----------------------------------------------------

        residual = NIRS_aligned - EEG_time_sec;

        ResidualMean(s) = mean(residual);
        ResidualStd(s) = std(residual);
        ResidualMax(s) = max(abs(residual));

        fprintf('\nOffsets:\n');
        fprintf('1-20  = %.4f sec\n',Offset1(s));
        fprintf('21-40 = %.4f sec\n',Offset2(s));
        fprintf('41-60 = %.4f sec\n',Offset3(s));

        fprintf('\nResidual:\n');
        fprintf('Mean = %.6f sec\n',ResidualMean(s));
        fprintf('Std  = %.6f sec\n',ResidualStd(s));
        fprintf('Max  = %.6f sec\n',ResidualMax(s));

    end

end

%% ============================================================
% CREATE SUMMARY TABLE
% ============================================================

Results = table( ...
    Subject, ...
    EEG_cnt, ...
    EEG_mrk, ...
    NIRS_cnt, ...
    NIRS_mrk, ...
    EEG_events, ...
    NIRS_events, ...
    ClassMatch, ...
    Offset1, ...
    Offset2, ...
    Offset3, ...
    ResidualMean, ...
    ResidualStd, ...
    ResidualMax);

%% ============================================================
% DISPLAY SUMMARY
% ============================================================

fprintf('\n\n============================================\n');
fprintf('BATCH VERIFICATION SUMMARY\n');
fprintf('============================================\n');

disp(Results);

%% ============================================================
% VALID SUBJECTS
% ============================================================

valid = ...
    EEG_cnt & ...
    EEG_mrk & ...
    NIRS_cnt & ...
    NIRS_mrk & ...
    EEG_events == 60 & ...
    NIRS_events == 60 & ...
    ClassMatch & ...
    ResidualMax < 0.1;

fprintf('\n============================================\n');
fprintf('VALID SUBJECTS\n');
fprintf('============================================\n');

fprintf('Valid subjects = %d / %d\n', ...
    sum(valid),nSubjects);

disp(Subject(valid));

%% ============================================================
% INVALID SUBJECTS
% ============================================================

fprintf('\n============================================\n');
fprintf('SUBJECTS REQUIRING INSPECTION\n');
fprintf('============================================\n');

disp(Subject(~valid));

%% ============================================================
% SAVE RESULT
% ============================================================

save(fullfile(basePath,"VF_batch_alignment_results.mat"), ...
    "Results");

fprintf('\nResults saved.\n');