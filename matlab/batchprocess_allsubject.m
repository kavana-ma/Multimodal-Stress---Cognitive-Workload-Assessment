%% ============================================================
% BATCH PREPROCESSING OF ALL PAIRED EEG + fNIRS SUBJECTS
%
% Purpose:
%   Apply the validated VP001 preprocessing pipeline to all
%   24 subjects having both synchronized EEG and fNIRS data.
%
% Input:
%   D:\major_project_group50\dataset\synchronized_epochs
%
% Output:
%   D:\major_project_group50\dataset\preprocessed_epochs
%
% Subjects:
%   VP001-VP011
%   VP014-VP026
%
% Preprocessing is performed by:
%   preprocess_subject.m
%
% IMPORTANT:
%   Do not change preprocessing parameters between subjects.
%
% ============================================================

clear;
clc;

%% ============================================================
% PATHS
% ============================================================

inputFolder = ...
    "D:\major_project_group50\dataset\synchronized_epochs";

outputFolder = ...
    "D:\major_project_group50\dataset\preprocessed_epochs";


%% ============================================================
% CHECK INPUT FOLDER
% ============================================================

if ~isfolder(inputFolder)

    error(['Input folder does not exist: ' ...
           char(inputFolder)]);

end


%% ============================================================
% CREATE OUTPUT FOLDER
% ============================================================

if ~isfolder(outputFolder)

    mkdir(outputFolder);

end


%% ============================================================
% PAIRED SUBJECTS
% ============================================================

subjects = [ ...
    "VP001"
    "VP002"
    "VP003"
    "VP004"
    "VP005"
    "VP006"
    "VP007"
    "VP008"
    "VP009"
    "VP010"
    "VP011"
    "VP014"
    "VP015"
    "VP016"
    "VP017"
    "VP018"
    "VP019"
    "VP020"
    "VP021"
    "VP022"
    "VP023"
    "VP024"
    "VP025"
    "VP026"
];


%% ============================================================
% INITIALIZE RESULTS
% ============================================================

nSubjects = numel(subjects);

success = false(nSubjects,1);

inputExists  = false(nSubjects,1);
outputExists = false(nSubjects,1);

errorMessage = strings(nSubjects,1);


%% ============================================================
% START BATCH PROCESSING
% ============================================================

fprintf('\n');
fprintf('============================================\n');
fprintf('BATCH EEG + fNIRS PREPROCESSING\n');
fprintf('============================================\n');

fprintf('Input folder:\n%s\n', inputFolder);

fprintf('\nOutput folder:\n%s\n', outputFolder);

fprintf('\nTotal paired subjects = %d\n', nSubjects);

fprintf('\nValidated preprocessing:\n');
fprintf('EEG  : 1-45 Hz, 28 EEG channels, baseline -5 to 0 s\n');
fprintf('fNIRS: 0.01-0.1 Hz, 36 channels, baseline -5 to 0 s\n');


%% ============================================================
% PROCESS EACH SUBJECT
% ============================================================

for s = 1:nSubjects

    subjectID = subjects(s);

    fprintf('\n');
    fprintf('============================================\n');
    fprintf('SUBJECT %s (%d/%d)\n', ...
        subjectID, s, nSubjects);
    fprintf('============================================\n');


    %% --------------------------------------------------------
    % INPUT FILE
    % ---------------------------------------------------------

    inputFile = fullfile( ...
        inputFolder, ...
        subjectID + "_synchronized_epochs.mat");


    %% --------------------------------------------------------
    % OUTPUT FILE
    % ---------------------------------------------------------

    outputFile = fullfile( ...
        outputFolder, ...
        subjectID + "_preprocessed.mat");


    %% --------------------------------------------------------
    % CHECK INPUT
    % ---------------------------------------------------------

    if ~isfile(inputFile)

        fprintf('INPUT FILE NOT FOUND\n');
        fprintf('%s\n', inputFile);

        errorMessage(s) = "Input file not found";

        continue;

    end

    inputExists(s) = true;

    fprintf('Input file found.\n');


    %% --------------------------------------------------------
    % PROCESS SUBJECT
    % ---------------------------------------------------------

    try

        fprintf('Running preprocessing...\n');

        preprocess_subject( ...
            subjectID, ...
            inputFolder, ...
            outputFolder);


        %% ----------------------------------------------------
        % CHECK OUTPUT
        % ----------------------------------------------------

        if isfile(outputFile)

            outputExists(s) = true;
            success(s) = true;

            fprintf('\nSTATUS = SUCCESS\n');
            fprintf('Output saved:\n');
            fprintf('%s\n', outputFile);

        else

            fprintf('\nSTATUS = FAILED\n');
            fprintf('Preprocessing finished but output file was not found.\n');

            errorMessage(s) = ...
                "Preprocessing completed but output file was not created";

        end


    catch ME

        fprintf('\nSTATUS = FAILED\n');

        fprintf('Error message:\n');
        fprintf('%s\n', ME.message);

        errorMessage(s) = string(ME.message);

    end

end


%% ============================================================
% FINAL SUMMARY
% ============================================================

fprintf('\n');
fprintf('============================================\n');
fprintf('BATCH PREPROCESSING SUMMARY\n');
fprintf('============================================\n');

fprintf('Total subjects      = %d\n', nSubjects);
fprintf('Input files found   = %d\n', sum(inputExists));
fprintf('Successful          = %d\n', sum(success));
fprintf('Failed              = %d\n', sum(~success));


%% ============================================================
% SUCCESSFUL SUBJECTS
% ============================================================

fprintf('\n');
fprintf('============================================\n');
fprintf('SUCCESSFUL SUBJECTS\n');
fprintf('============================================\n');

successfulSubjects = subjects(success);

if isempty(successfulSubjects)

    fprintf('None.\n');

else

    for s = 1:numel(successfulSubjects)

        fprintf('%d. %s\n', ...
            s, successfulSubjects(s));

    end

end


%% ============================================================
% FAILED SUBJECTS
% ============================================================

fprintf('\n');
fprintf('============================================\n');
fprintf('FAILED SUBJECTS\n');
fprintf('============================================\n');

failedSubjects = subjects(~success);

if isempty(failedSubjects)

    fprintf('None.\n');

else

    for s = 1:numel(failedSubjects)

        idx = find(subjects == failedSubjects(s),1);

        fprintf('%d. %s\n', ...
            s, failedSubjects(s));

        fprintf('   Reason: %s\n', ...
            errorMessage(idx));

    end

end


%% ============================================================
% VERIFY OUTPUT FILES
% ============================================================

fprintf('\n');
fprintf('============================================\n');
fprintf('OUTPUT FILE VERIFICATION\n');
fprintf('============================================\n');

verifiedCount = 0;

for s = 1:nSubjects

    outputFile = fullfile( ...
        outputFolder, ...
        subjects(s) + "_preprocessed.mat");

    if isfile(outputFile)

        verifiedCount = verifiedCount + 1;

        fprintf('[OK] %s\n', ...
            subjects(s));

    else

        fprintf('[MISSING] %s\n', ...
            subjects(s));

    end

end


%% ============================================================
% FINAL RESULT
% ============================================================

fprintf('\n');
fprintf('============================================\n');
fprintf('BATCH PREPROCESSING COMPLETE\n');
fprintf('============================================\n');

fprintf('Expected subjects = %d\n', nSubjects);
fprintf('Verified outputs  = %d\n', verifiedCount);

if verifiedCount == nSubjects

    fprintf('\nALL 24 SUBJECTS SUCCESSFULLY PREPROCESSED.\n');

else

    fprintf('\nWARNING: NOT ALL SUBJECTS WERE PREPROCESSED.\n');

end

fprintf('\nOutput folder:\n');
fprintf('%s\n', outputFolder);

fprintf('\n============================================\n');