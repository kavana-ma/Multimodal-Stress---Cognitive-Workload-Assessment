%% ============================================================
% FIND EEG-fNIRS SUBJECT PAIRS
% ============================================================

clear;
clc;

% ------------------------------------------------------------
% MAIN DATASET FOLDER
% ------------------------------------------------------------

dataset_root = "D:\major_project_group50\dataset";

% ------------------------------------------------------------
% FIND ALL EEG AND NIRS FOLDERS
% ------------------------------------------------------------

all_folders = dir(dataset_root);

EEG_subjects  = {};
NIRS_subjects = {};

for i = 1:length(all_folders)

    if ~all_folders(i).isdir
        continue;
    end

    folder_name = all_folders(i).name;

    % Ignore "." and ".."
    if strcmp(folder_name,'.') || strcmp(folder_name,'..')
        continue;
    end

    % EEG folder
    token = regexp(folder_name, '^(VP\d+)-EEG$', 'tokens');

    if ~isempty(token)
        EEG_subjects{end+1} = token{1}{1};
    end

    % NIRS folder
    token = regexp(folder_name, '^(VP\d+)-NIRS$', 'tokens');

    if ~isempty(token)
        NIRS_subjects{end+1} = token{1}{1};
    end

end

% ------------------------------------------------------------
% FIND COMMON SUBJECTS
% ------------------------------------------------------------

paired_subjects = intersect(EEG_subjects, NIRS_subjects);

% Sort naturally by subject number
subject_numbers = cellfun(@(x) str2double(extractAfter(x,2)), ...
    paired_subjects);

[~, order] = sort(subject_numbers);

paired_subjects = paired_subjects(order);

% ------------------------------------------------------------
% DISPLAY RESULTS
% ------------------------------------------------------------

fprintf('\n========== SUBJECT SUMMARY ==========\n');

fprintf('EEG subjects found  = %d\n', length(EEG_subjects));
fprintf('NIRS subjects found = %d\n', length(NIRS_subjects));
fprintf('Paired subjects     = %d\n', length(paired_subjects));

fprintf('\n========== PAIRED SUBJECTS ==========\n');

for i = 1:length(paired_subjects)

    fprintf('%2d : %s\n', i, paired_subjects{i});

end