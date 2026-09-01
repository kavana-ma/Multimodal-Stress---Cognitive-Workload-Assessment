%% ============================================================
% CHECK REQUIRED FILES FOR EVERY PAIRED SUBJECT
% ============================================================

fprintf('\n========== FILE CHECK ==========\n');

valid_subjects = {};

for i = 1:length(paired_subjects)

    subject = paired_subjects{i};

    eeg_folder  = fullfile(dataset_root, subject + "-EEG");
    nirs_folder = fullfile(dataset_root, subject + "-NIRS");

    % Required files
    eeg_cnt = fullfile(eeg_folder, "cnt_vf.mat");
    eeg_mrk = fullfile(eeg_folder, "mrk_vf.mat");

    nirs_cnt = fullfile(nirs_folder, "cnt_vf.mat");
    nirs_mrk = fullfile(nirs_folder, "mrk_vf.mat");

    files_exist = ...
        isfile(eeg_cnt) && ...
        isfile(eeg_mrk) && ...
        isfile(nirs_cnt) && ...
        isfile(nirs_mrk);

    if files_exist

        fprintf('%s : OK\n', subject);

        valid_subjects{end+1} = subject;

    else

        fprintf('%s : MISSING FILE(S)\n', subject);

        if ~isfile(eeg_cnt)
            fprintf('       Missing EEG cnt_vf.mat\n');
        end

        if ~isfile(eeg_mrk)
            fprintf('       Missing EEG mrk_vf.mat\n');
        end

        if ~isfile(nirs_cnt)
            fprintf('       Missing NIRS cnt_vf.mat\n');
        end

        if ~isfile(nirs_mrk)
            fprintf('       Missing NIRS mrk_vf.mat\n');
        end

    end

end

fprintf('\nValid paired subjects = %d\n', length(valid_subjects));