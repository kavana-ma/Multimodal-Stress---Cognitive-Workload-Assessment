%% ============================================================
% FIND PAIRED EEG-NIRS SUBJECTS
% ============================================================

clear;
clc;

rootDir = "D:\major_project_group50\dataset";

%% Get all folders

allFolders = dir(rootDir);

allFolders = allFolders([allFolders.isdir]);

% Remove . and ..
allFolders = allFolders(~ismember({allFolders.name}, {'.','..'}));

%% ============================================================
% FIND EEG AND NIRS FOLDERS
% ============================================================

EEG_folders  = {};
NIRS_folders = {};

for i = 1:length(allFolders)

    folderName = allFolders(i).name;

    if endsWith(folderName, '-EEG')
        EEG_folders{end+1} = folderName;
    end

    if endsWith(folderName, '-NIRS')
        NIRS_folders{end+1} = folderName;
    end

end

fprintf('\n============================================\n');
fprintf('EEG FOLDERS FOUND\n');
fprintf('============================================\n');

disp(EEG_folders');

fprintf('\n============================================\n');
fprintf('NIRS FOLDERS FOUND\n');
fprintf('============================================\n');

disp(NIRS_folders');

%% ============================================================
% EXTRACT SUBJECT IDs
% ============================================================

EEG_subjects = erase(EEG_folders, '-EEG');
NIRS_subjects = erase(NIRS_folders, '-NIRS');

%% ============================================================
% FIND COMMON SUBJECTS
% ============================================================

pairedSubjects = intersect(EEG_subjects, NIRS_subjects);

fprintf('\n============================================\n');
fprintf('PAIRED SUBJECTS FOUND\n');
fprintf('============================================\n');

fprintf('Total paired subjects = %d\n\n', length(pairedSubjects));

for i = 1:length(pairedSubjects)

    fprintf('%d. %s\n', i, pairedSubjects{i});

end

%% ============================================================
% FIND EEG-ONLY SUBJECTS
% ============================================================

EEG_only = setdiff(EEG_subjects, NIRS_subjects);

fprintf('\n============================================\n');
fprintf('EEG ONLY SUBJECTS\n');
fprintf('============================================\n');

fprintf('Count = %d\n', length(EEG_only));

disp(EEG_only');

%% ============================================================
% FIND NIRS-ONLY SUBJECTS
% ============================================================

NIRS_only = setdiff(NIRS_subjects, EEG_subjects);

fprintf('\n============================================\n');
fprintf('NIRS ONLY SUBJECTS\n');
fprintf('============================================\n');

fprintf('Count = %d\n', length(NIRS_only));

disp(NIRS_only');