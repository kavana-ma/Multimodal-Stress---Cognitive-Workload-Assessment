%% ============================================================
% VP001 PREPROCESSING VALIDATION
% ============================================================

fprintf('\n============================================\n');
fprintf('VP001 PREPROCESSING VALIDATION\n');
fprintf('============================================\n');

%% ------------------------------------------------------------
% 1. CHECK VARIABLES
% ------------------------------------------------------------

fprintf('\n========== AVAILABLE VARIABLES ==========\n');
whos

%% ------------------------------------------------------------
% 2. SET PREPROCESSED VARIABLES
% ------------------------------------------------------------
% Change these three names ONLY if your preprocessing script
% uses different output variable names.

EEGp   = EEG_preprocessed;
Oxyp   = Oxy_preprocessed;
Deoxyp = Deoxy_preprocessed;

fprintf('\n========== PREPROCESSED DATA ==========\n');

fprintf('EEG   : %d x %d x %d\n', size(EEGp,1), size(EEGp,2), size(EEGp,3));
fprintf('Oxy   : %d x %d x %d\n', size(Oxyp,1), size(Oxyp,2), size(Oxyp,3));
fprintf('Deoxy : %d x %d x %d\n', size(Deoxyp,1), size(Deoxyp,2), size(Deoxyp,3));

%% ------------------------------------------------------------
% 3. CLASS INDICES
% ------------------------------------------------------------

BL_idx = find(eventClass == "BL");
VF_idx = find(eventClass == "VF");

fprintf('\n========== CLASS COUNTS ==========\n');

fprintf('BL trials = %d\n', length(BL_idx));
fprintf('VF trials = %d\n', length(VF_idx));

%% ------------------------------------------------------------
% 4. TIME AXES
% ------------------------------------------------------------

eeg_t  = EEG_time_axis;
nirs_t = NIRS_time_axis;

fprintf('\n========== TIME WINDOWS ==========\n');

fprintf('EEG epoch   : %.2f to %.2f sec\n', ...
    eeg_t(1), eeg_t(end));

fprintf('NIRS epoch  : %.2f to %.2f sec\n', ...
    nirs_t(1), nirs_t(end));

%% ------------------------------------------------------------
% 5. EEG BASELINE VALIDATION
% ------------------------------------------------------------

eeg_baseline = eeg_t < 0;

EEG_baseline_mean = squeeze(mean(EEGp(eeg_baseline,:,:),1));

fprintf('\n========== EEG BASELINE ==========\n');

fprintf('Mean absolute baseline = %.10f uV\n', ...
    mean(abs(EEG_baseline_mean(:))));

fprintf('Maximum absolute baseline = %.10f uV\n', ...
    max(abs(EEG_baseline_mean(:))));

%% ------------------------------------------------------------
% 6. NIRS BASELINE VALIDATION
% ------------------------------------------------------------

nirs_baseline = nirs_t < 0;

Oxy_baseline_mean = squeeze(mean(Oxyp(nirs_baseline,:,:),1));

Deoxy_baseline_mean = squeeze(mean(Deoxyp(nirs_baseline,:,:),1));

fprintf('\n========== NIRS BASELINE ==========\n');

fprintf('Oxy mean absolute baseline = %.10f\n', ...
    mean(abs(Oxy_baseline_mean(:))));

fprintf('Deoxy mean absolute baseline = %.10f\n', ...
    mean(abs(Deoxy_baseline_mean(:))));

%% ------------------------------------------------------------
% 7. EEG TRIAL AMPLITUDE
% ------------------------------------------------------------

eeg_post = eeg_t >= 0 & eeg_t <= 15;

EEG_trial_mean = squeeze(mean(EEGp(eeg_post,:,:),1));

BL_EEG = EEG_trial_mean(:,BL_idx);
VF_EEG = EEG_trial_mean(:,VF_idx);

fprintf('\n========== EEG POST-STIMULUS ==========\n');

fprintf('BL grand mean = %.4f uV\n', mean(BL_EEG(:)));
fprintf('VF grand mean = %.4f uV\n', mean(VF_EEG(:)));

fprintf('BL grand std  = %.4f uV\n', std(BL_EEG(:)));
fprintf('VF grand std  = %.4f uV\n', std(VF_EEG(:)));

%% ------------------------------------------------------------
% 8. OXY POST-STIMULUS
% ------------------------------------------------------------

nirs_post = nirs_t >= 0 & nirs_t <= 15;

Oxy_trial_mean = squeeze(mean(Oxyp(nirs_post,:,:),1));

BL_Oxy = Oxy_trial_mean(:,BL_idx);
VF_Oxy = Oxy_trial_mean(:,VF_idx);

fprintf('\n========== OXY POST-STIMULUS ==========\n');

fprintf('BL grand mean = %.8f\n', mean(BL_Oxy(:)));
fprintf('VF grand mean = %.8f\n', mean(VF_Oxy(:)));

fprintf('BL grand std  = %.8f\n', std(BL_Oxy(:)));
fprintf('VF grand std  = %.8f\n', std(VF_Oxy(:)));

%% ------------------------------------------------------------
% 9. DEOXY POST-STIMULUS
% ------------------------------------------------------------

Deoxy_trial_mean = squeeze(mean(Deoxyp(nirs_post,:,:),1));

BL_Deoxy = Deoxy_trial_mean(:,BL_idx);
VF_Deoxy = Deoxy_trial_mean(:,VF_idx);

fprintf('\n========== DEOXY POST-STIMULUS ==========\n');

fprintf('BL grand mean = %.8f\n', mean(BL_Deoxy(:)));
fprintf('VF grand mean = %.8f\n', mean(VF_Deoxy(:)));

fprintf('BL grand std  = %.8f\n', std(BL_Deoxy(:)));
fprintf('VF grand std  = %.8f\n', std(VF_Deoxy(:)));

%% ------------------------------------------------------------
% 10. CHECK FOR NaN / INF
% ------------------------------------------------------------

fprintf('\n========== FINAL DATA QUALITY ==========\n');

fprintf('EEG NaN   = %d\n', sum(isnan(EEGp(:))));
fprintf('EEG Inf   = %d\n', sum(isinf(EEGp(:))));

fprintf('Oxy NaN   = %d\n', sum(isnan(Oxyp(:))));
fprintf('Oxy Inf   = %d\n', sum(isinf(Oxyp(:))));

fprintf('Deoxy NaN = %d\n', sum(isnan(Deoxyp(:))));
fprintf('Deoxy Inf = %d\n', sum(isinf(Deoxyp(:))));

%% ------------------------------------------------------------
% 11. SIMPLE BL vs VF VISUALIZATION
% ------------------------------------------------------------

figure;

subplot(3,1,1);

plot(eeg_t, mean(EEGp(:,:,BL_idx),3));
hold on;
plot(eeg_t, mean(EEGp(:,:,VF_idx),3));

xline(0,'--');

xlabel('Time (s)');
ylabel('EEG (\muV)');
title('VP001 EEG: BL vs VF');
legend('BL','VF');
grid on;


subplot(3,1,2);

plot(nirs_t, mean(Oxyp(:,:,BL_idx),3));
hold on;
plot(nirs_t, mean(Oxyp(:,:,VF_idx),3));

xline(0,'--');

xlabel('Time (s)');
ylabel('\DeltaHbO');
title('VP001 HbO: BL vs VF');
legend('BL','VF');
grid on;


subplot(3,1,3);

plot(nirs_t, mean(Deoxyp(:,:,BL_idx),3));
hold on;
plot(nirs_t, mean(Deoxyp(:,:,VF_idx),3));

xline(0,'--');

xlabel('Time (s)');
ylabel('\DeltaHbR');
title('VP001 HbR: BL vs VF');
legend('BL','VF');
grid on;

fprintf('\n============================================\n');
fprintf('VP001 PREPROCESSING VALIDATION COMPLETE\n');
fprintf('============================================\n');