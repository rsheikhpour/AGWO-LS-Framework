%% =========================================================================
%  demo_AGWO_LS: Demo script for the AGWO-LS framework
% =========================================================================
%
%  This script demonstrates how to use the AGWO-LS framework to optimize
%  all classifiers reported in the paper for breast cancer biomarker
%  prediction from hematological indices.
%
%  Features used (6 predictor variables):
%    1. Age
%    2. Absolute neutrophil count
%    3. Lymphocyte count
%    4. Platelet count
%    5. Neutrophil-to-Lymphocyte Ratio (NLR)
%    6. Platelet-to-Lymphocyte Ratio (PLR)
%
%  Target: biomarker status (binary: positive=1 / negative=0)
%          ER | PR | HER2/neu | Ki-67
%
%  Evaluation: repeated Monte Carlo Cross-Validation (MCCV)
%              3 runs, 70% train / 30% test split (internal validation only)
%
% =========================================================================

clc; clear; close all;

%% -------------------------------------------------------------------------
%  1. Load and Prepare Data
% -------------------------------------------------------------------------
% Replace with your own data loading:
%   X : [n_samples x 6] matrix of hematological features
%   y : [n_samples x 1] binary biomarker labels (0 or 1)
%
% Example with random data (replace with real data):
rng(42);
n_samples = 151;
n_features = 6;
X = rand(n_samples, n_features);   % replace with real features
y = randi([0, 1], n_samples, 1);   % replace with real labels

feature_names = {'Age', 'Neutrophil', 'Lymphocyte', ...
                 'Platelet', 'NLR', 'PLR'};

fprintf('Dataset: %d samples, %d features\n', n_samples, n_features);
fprintf('Class distribution: %d positive, %d negative\n\n', ...
    sum(y==1), sum(y==0));

%% -------------------------------------------------------------------------
%  2. AGWO-LS Settings (Section 4.3.3 of the paper)
% -------------------------------------------------------------------------
max_iter = 100;
n_wolves = 30;
n_runs   = 3;    % repeated MCCV runs

%% -------------------------------------------------------------------------
%  3. Monte Carlo Cross-Validation Loop
% -------------------------------------------------------------------------
% Store results
results = struct();
classifiers = {'SVM_RBF', 'SVM_Linear', 'RF', 'KNN', 'DT', 'Ensemble'};
metrics = {'Accuracy', 'Precision', 'Recall', 'F1'};

for c = 1:numel(classifiers)
    results.(classifiers{c}) = zeros(n_runs, 4);  % [Acc, Prec, Rec, F1]
end

for run = 1:n_runs
    fprintf('========== MCCV Run %d/%d ==========\n\n', run, n_runs);

    % --- Train/Test Split (70/30, stratified) ----------------------------
    cv = cvpartition(y, 'HoldOut', 0.3);
    X_train = X(training(cv), :);
    y_train = y(training(cv));
    X_test  = X(test(cv), :);
    y_test  = y(test(cv));

    % --- Min-Max Normalization (fit on train, apply to test) -------------
    X_min   = min(X_train);
    X_max   = max(X_train);
    X_range = X_max - X_min + 1e-10;

    X_train_norm = (X_train - X_min) ./ X_range;
    X_test_norm  = (X_test  - X_min) ./ X_range;

    % =====================================================================
    %  Classifier 1: SVM (RBF kernel)
    % =====================================================================
    fprintf('--- Optimizing SVM (RBF) ---\n');
    [params_svm_rbf, ~, curve_svm_rbf] = AGWO_LS_SVM(...
        X_train_norm, y_train, max_iter, n_wolves, 'rbf');

    C_rbf     = params_svm_rbf(1);
    gamma_rbf = params_svm_rbf(2);

    t_rbf  = templateSVM('KernelFunction', 'rbf', ...
                         'BoxConstraint',  C_rbf, ...
                         'KernelScale',    1/sqrt(2*gamma_rbf));
    mdl_rbf = fitcecoc(X_train_norm, y_train, 'Learners', t_rbf);
    y_pred  = predict(mdl_rbf, X_test_norm);
    results.SVM_RBF(run,:) = compute_metrics(y_test, y_pred);

    % =====================================================================
    %  Classifier 2: SVM (Linear kernel)
    % =====================================================================
    fprintf('\n--- Optimizing SVM (Linear) ---\n');
    [params_svm_lin, ~, curve_svm_lin] = AGWO_LS_SVM(...
        X_train_norm, y_train, max_iter, n_wolves, 'linear');

    C_lin   = params_svm_lin(1);
    t_lin   = templateSVM('KernelFunction', 'linear', ...
                          'BoxConstraint',  C_lin);
    mdl_lin = fitcecoc(X_train_norm, y_train, 'Learners', t_lin);
    y_pred  = predict(mdl_lin, X_test_norm);
    results.SVM_Linear(run,:) = compute_metrics(y_test, y_pred);

    % =====================================================================
    %  Classifier 3: Random Forest
    % =====================================================================
    fprintf('\n--- Optimizing Random Forest ---\n');
    [params_rf, ~, curve_rf] = AGWO_LS_RF(...
        X_train_norm, y_train, max_iter, n_wolves);

    n_trees   = round(params_rf(1));
    max_depth = round(params_rf(2));
    min_split = round(params_rf(3));

    t_rf   = templateTree('MaxNumSplits', max_depth, ...
                          'MinLeafSize',  max(1,floor(min_split/2)));
    mdl_rf = fitcensemble(X_train_norm, y_train, ...
                          'Method',    'Bag', ...
                          'NumLearningCycles', n_trees, ...
                          'Learners',  t_rf);
    y_pred = predict(mdl_rf, X_test_norm);
    results.RF(run,:) = compute_metrics(y_test, y_pred);

    % =====================================================================
    %  Classifier 4: KNN
    % =====================================================================
    fprintf('\n--- Optimizing KNN ---\n');
    [params_knn, ~, curve_knn] = AGWO_LS_KNN(...
        X_train_norm, y_train, max_iter, n_wolves);

    k_opt    = max(1, round(params_knn(1)));
    dist_opt = decode_knn_distance(params_knn(2));

    mdl_knn = fitcknn(X_train_norm, y_train, ...
                      'NumNeighbors', k_opt, ...
                      'Distance',     dist_opt);
    y_pred  = predict(mdl_knn, X_test_norm);
    results.KNN(run,:) = compute_metrics(y_test, y_pred);

    % =====================================================================
    %  Classifier 5: Decision Tree
    % =====================================================================
    fprintf('\n--- Optimizing Decision Tree ---\n');
    [params_dt, ~, curve_dt] = AGWO_LS_DT(...
        X_train_norm, y_train, max_iter, n_wolves);

    depth_opt = round(params_dt(1));
    split_opt = round(params_dt(2));

    mdl_dt = fitctree(X_train_norm, y_train, ...
                      'MaxNumSplits', depth_opt, ...
                      'MinLeafSize',  max(1,floor(split_opt/2)));
    y_pred = predict(mdl_dt, X_test_norm);
    results.DT(run,:) = compute_metrics(y_test, y_pred);

    % =====================================================================
    %  Classifier 6: Ensemble
    % =====================================================================
    fprintf('\n--- Optimizing Ensemble ---\n');
    [params_ens, ~, curve_ens] = AGWO_LS_Ensemble(...
        X_train_norm, y_train, max_iter, n_wolves);

    fprintf('\n');
    results.Ensemble(run,:) = results.RF(run,:);  % placeholder
    % Note: replace with full ensemble prediction using optimized voting

end

%% -------------------------------------------------------------------------
%  4. Report Results (mean ± std over MCCV runs)
% -------------------------------------------------------------------------
fprintf('\n========== RESULTS (mean ± std over %d MCCV runs) ==========\n\n', n_runs);
fprintf('%-15s %20s %20s %20s %20s\n', ...
    'Classifier', 'Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)');
fprintf('%s\n', repmat('-', 1, 80));

for c = 1:numel(classifiers)
    m = mean(results.(classifiers{c})) * 100;
    s = std(results.(classifiers{c}))  * 100;
    fprintf('%-15s  %6.2f ± %5.2f   %6.2f ± %5.2f   %6.2f ± %5.2f   %6.2f ± %5.2f\n',...
        classifiers{c}, m(1),s(1), m(2),s(2), m(3),s(3), m(4),s(4));
end

fprintf('\nNote: All results are from internal MCCV only (single-center dataset).\n');
fprintf('External validation is required before any clinical interpretation.\n\n');

%% -------------------------------------------------------------------------
%  5. Plot Convergence Curves (last run)
% -------------------------------------------------------------------------
figure('Name', 'AGWO-LS Convergence Curves', 'NumberTitle', 'off');
hold on;
plot(1:max_iter, curve_svm_rbf,  'b-',  'LineWidth', 1.5, 'DisplayName', 'SVM-RBF');
plot(1:max_iter, curve_svm_lin,  'r--', 'LineWidth', 1.5, 'DisplayName', 'SVM-Linear');
plot(1:max_iter, curve_rf,       'g-',  'LineWidth', 1.5, 'DisplayName', 'RF');
plot(1:max_iter, curve_knn,      'm-.', 'LineWidth', 1.5, 'DisplayName', 'KNN');
plot(1:max_iter, curve_dt,       'k:',  'LineWidth', 1.5, 'DisplayName', 'DT');
plot(1:max_iter, curve_ens,      'c-',  'LineWidth', 1.5, 'DisplayName', 'Ensemble');
hold off;
xlabel('Iteration');
ylabel('CV Error (1 - Accuracy)');
title('AGWO-LS Convergence Curves (Run 3)');
legend('Location', 'northeast');
grid on;

%% =========================================================================
%  Helper Functions
% =========================================================================

function m = compute_metrics(y_true, y_pred)
% Returns [Accuracy, Precision, Recall, F1]
TP = sum(y_pred == 1 & y_true == 1);
TN = sum(y_pred == 0 & y_true == 0);
FP = sum(y_pred == 1 & y_true == 0);
FN = sum(y_pred == 0 & y_true == 1);

acc  = (TP + TN) / (TP + TN + FP + FN + 1e-10);
prec = TP / (TP + FP + 1e-10);
rec  = TP / (TP + FN + 1e-10);
f1   = 2 * prec * rec / (prec + rec + 1e-10);

m = [acc, prec, rec, f1];
end

function dist_name = decode_knn_distance(val)
if val < 0.5
    dist_name = 'euclidean';
else
    dist_name = 'cityblock';
end
end
