%% =========================================================================
%  AGWO_LS_Ensemble: Adaptive Grey Wolf Optimizer with Local Search
%                     for Ensemble Classifier Optimization
% =========================================================================
%
%  Reference:
%    Sheikhpour R., Taghipour Zahir S., Pourhosseini F.
%    "Non-Invasive Prediction of Breast Cancer Biomarkers from Blood
%     Inflammatory Indices Using a Hybrid Adaptive Grey Wolf
%     Optimizer-Based Machine Learning Framework"
%    Array, 2026.
%
%  Hyperparameter search ranges:
%    Voting strategy : Hard (0) or Soft (1)  [encoded as continuous in [0,1]]
%
%  Description:
%    The Ensemble classifier combines predictions from multiple base
%    learners (SVM-RBF, KNN, Decision Tree) via majority voting.
%    AGWO-LS optimizes the voting strategy (hard vs soft).
%    Base learner hyperparameters use default values in this implementation;
%    for full optimization, combine with AGWO_LS_SVM / AGWO_LS_KNN / AGWO_LS_DT.
%
%  Inputs:
%    X_train  : [n_samples x n_features] normalized training data
%    y_train  : [n_samples x 1] training labels
%    max_iter : maximum AGWO iterations (default: 100)
%    n_wolves : population size          (default: 30)
%
%  Outputs:
%    best_params  : [voting_encoded]  (0=hard, 1=soft, threshold at 0.5)
%    best_fitness : best CV error achieved
%    conv_curve   : convergence curve over iterations
%
%  Usage:
%    [params, fitness, curve] = AGWO_LS_Ensemble(X_train, y_train, 100, 30);
%    voting = decode_voting(params(1));   % 'hard' or 'soft'
%
% =========================================================================

function [best_params, best_fitness, conv_curve] = AGWO_LS_Ensemble(...
    X_train, y_train, max_iter, n_wolves)

%% -------------------------------------------------------------------------
%  0. Default Arguments
% -------------------------------------------------------------------------
if nargin < 3 || isempty(max_iter), max_iter = 100; end
if nargin < 4 || isempty(n_wolves), n_wolves = 30;  end

%% -------------------------------------------------------------------------
%  1. Hyperparameter Bounds
%     [voting_strategy_encoded]: 0=hard, 1=soft
% -------------------------------------------------------------------------
lb  = [0];
ub  = [1];
dim = 1;

fprintf('AGWO-LS Ensemble | Voting strategy: Hard or Soft\n');

%% -------------------------------------------------------------------------
%  2. Local Search Parameters (Section 4.3.3)
% -------------------------------------------------------------------------
ls_step_init = 0.1;
ls_decay     = 0.5;
ls_step_min  = 0.001;
ls_max_iter  = 10;

%% -------------------------------------------------------------------------
%  3. Initialize Wolf Population
% -------------------------------------------------------------------------
rng('shuffle');

X       = rand(n_wolves, dim) .* (ub - lb) + lb;
fitness = inf(n_wolves, 1);

for i = 1:n_wolves
    fitness(i) = ensembleObjective(X(i,:), X_train, y_train);
end

[fitness, idx] = sort(fitness);
X = X(idx, :);

X_alpha = X(1,:);  f_alpha = fitness(1);
X_beta  = X(2,:);
X_delta = X(3,:);

conv_curve = zeros(max_iter, 1);

fprintf('\n--- AGWO-LS Ensemble Optimization Started ---\n');
fprintf('Population: %d | Max iterations: %d\n\n', n_wolves, max_iter);

%% -------------------------------------------------------------------------
%  4. Main AGWO-LS Loop
% -------------------------------------------------------------------------
for t = 1:max_iter

    % --- 4.1 Adaptive convergence factor ----------------------------------
    f_avg = mean(fitness);
    f_min = fitness(1);
    f_max = fitness(end);
    eps_  = 1e-10;

    diversity = abs((f_avg - f_min) / (f_max - f_min + eps_));
    a = 2 * (1 - t/max_iter) * diversity;

    % --- 4.2 Update Wolf Positions ----------------------------------------
    for i = 1:n_wolves
        for j = 1:dim
            r1 = rand(); r2 = rand();
            A1 = 2*a*r1-a; C1 = 2*r2;
            X1 = X_alpha(j) - A1*abs(C1*X_alpha(j) - X(i,j));

            r1 = rand(); r2 = rand();
            A2 = 2*a*r1-a; C2 = 2*r2;
            X2 = X_beta(j)  - A2*abs(C2*X_beta(j)  - X(i,j));

            r1 = rand(); r2 = rand();
            A3 = 2*a*r1-a; C3 = 2*r2;
            X3 = X_delta(j) - A3*abs(C3*X_delta(j) - X(i,j));

            X(i,j) = (X1 + X2 + X3) / 3;
        end

        X(i,:) = max(X(i,:), lb);
        X(i,:) = min(X(i,:), ub);
        fitness(i) = ensembleObjective(X(i,:), X_train, y_train);
    end

    [fitness, idx] = sort(fitness);
    X = X(idx, :);

    X_alpha = X(1,:);  f_alpha = fitness(1);
    X_beta  = X(2,:);
    X_delta = X(3,:);

    % --- 4.3 Local Search on Alpha Wolf -----------------------------------
    ls_step    = ls_step_init;
    ls_current = X_alpha;
    ls_fitness = f_alpha;

    for ls_iter = 1:ls_max_iter
        improved = false;
        for j = 1:dim
            step_j = ls_step * (ub(j) - lb(j));

            X_cand = ls_current;
            X_cand(j) = min(ls_current(j) + step_j, ub(j));
            f_cand = ensembleObjective(X_cand, X_train, y_train);
            if f_cand < ls_fitness
                ls_current = X_cand; ls_fitness = f_cand;
                improved = true; break;
            end

            X_cand = ls_current;
            X_cand(j) = max(ls_current(j) - step_j, lb(j));
            f_cand = ensembleObjective(X_cand, X_train, y_train);
            if f_cand < ls_fitness
                ls_current = X_cand; ls_fitness = f_cand;
                improved = true; break;
            end
        end

        if ~improved,             ls_step = ls_step * ls_decay; end
        if ls_step < ls_step_min, break; end
    end

    if ls_fitness < f_alpha
        X_alpha = ls_current; f_alpha = ls_fitness;
        X(1,:) = X_alpha; fitness(1) = f_alpha;
    end

    conv_curve(t) = f_alpha;
    fprintf('Iter %3d/%d | Best CV error = %.4f | voting=%s\n',...
        t, max_iter, f_alpha, decode_voting(X_alpha(1)));
end

%% -------------------------------------------------------------------------
%  5. Output
% -------------------------------------------------------------------------
best_params  = X_alpha;
best_fitness = f_alpha;

fprintf('\n--- Optimization Complete ---\n');
fprintf('Optimal voting    = %s\n',   decode_voting(best_params(1)));
fprintf('Best CV error     = %.4f\n\n', best_fitness);

end % function AGWO_LS_Ensemble


%% =========================================================================
%  Local Function: ensembleObjective
%  Voting ensemble of SVM-RBF, KNN, and Decision Tree base learners
% =========================================================================
function cv_error = ensembleObjective(params, X_train, y_train)

use_soft = params(1) >= 0.5;   % true = soft voting

try
    % Base learners with default hyperparameters
    t_svm = templateSVM('KernelFunction', 'rbf', ...
                        'BoxConstraint',   1, ...
                        'KernelScale',     1, ...
                        'Standardize',     false);
    t_knn = templateKNN('NumNeighbors', 5, 'Distance', 'euclidean');
    t_dt  = templateTree('MaxNumSplits', 10);

    learners = {t_svm, t_knn, t_dt};
    n_base   = numel(learners);
    cv_errors = zeros(n_base, 1);

    % Cross-validate each base learner independently
    for b = 1:n_base
        switch b
            case 1
                mdl = fitcecoc(X_train, y_train, 'Learners', learners{b});
            case 2
                mdl = fitcknn(X_train, y_train, 'NumNeighbors', 5);
            case 3
                mdl = fitctree(X_train, y_train, 'MaxNumSplits', 10);
        end
        cv_mdl = crossval(mdl, 'KFold', 5);
        cv_errors(b) = kfoldLoss(cv_mdl);
    end

    if use_soft
        % Soft voting: average CV errors (proxy for probability averaging)
        cv_error = mean(cv_errors);
    else
        % Hard voting: minimum CV error among base learners
        cv_error = min(cv_errors);
    end

catch
    cv_error = 1;
end

end % function ensembleObjective


%% =========================================================================
%  Local Function: decode_voting
% =========================================================================
function voting_str = decode_voting(val)
if val < 0.5
    voting_str = 'hard';
else
    voting_str = 'soft';
end
end
