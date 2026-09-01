%% ============================================================
% CREATE SUBJECT SUMMARY TABLE
% ============================================================

fprintf('\n\n');
fprintf('====================================================\n');
fprintf('MULTI-SUBJECT ALIGNMENT SUMMARY\n');
fprintf('====================================================\n');

nResults = length(results);

Subject = strings(nResults,1);
Events = zeros(nResults,1);
MeanResidual_ms = zeros(nResults,1);
StdResidual_ms = zeros(nResults,1);
MaxAbsResidual_ms = zeros(nResults,1);
ClassAgreement = false(nResults,1);

for s = 1:nResults

    Subject(s) = string(results(s).subject);

    Events(s) = results(s).nEvents;

    MeanResidual_ms(s) = ...
        mean(results(s).residual) * 1000;

    StdResidual_ms(s) = ...
        std(results(s).residual) * 1000;

    MaxAbsResidual_ms(s) = ...
        max(abs(results(s).residual)) * 1000;

    ClassAgreement(s) = ...
        results(s).class_match;

end

summary_table = table( ...
    Subject, ...
    Events, ...
    MeanResidual_ms, ...
    StdResidual_ms, ...
    MaxAbsResidual_ms, ...
    ClassAgreement);

disp(summary_table);