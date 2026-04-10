%% =========================================================================
%  AGWO_LS_KNN: Adaptive Grey Wolf Optimizer with Local Search (AGWO-LS)
%                for k-Nearest Neighbors Hyperparameter Optimization
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
%    Number of neighbors k : [1, 20]
%    Distance metric       : Euclidean (0) or Manhattan (1)  [encoded]
%
%  Note on distance metric encoding:
%    The distance metric is a categorical parameter encoded as a
%    continuous value in [0,1] and thresholded at 0.5:
%      value < 0.5  -> 'euclidean'
%      value >= 0.5 -> 'cityblock' (Manhattan)
%
%  Inputs:
%    X_train  : [n_samples x n_features] normalized training data
%    y_train  : [n_samples x 1] training labels
%    max_iter : maximum AGWO iterations (default: 100)
%    n_wolves : population size          (default: 30)
%
%  Outputs:
%    best_params  : [k, dist_encoded]
%    best_fitness : best CV error achieved
%    conv_curve   : convergence curve over iterations
%
%  Usage:
%    [params, fitness, curve] = AGWO_LS_KNN(X_train, y_train, 100, 30);
%    k    = round(params(1));
%    dist = decode_distance(params(2));  % 'euclidean' or 'cityblock'
%
% =========================================================================

function [best_params, best_fitness, conv_curve] = AGWO_LS_KNN(...
    X_train, y_train, max_iter, n_wolves)

%% -------------------------------------------------------------------------
%  0. Default Arguments
% -------------------------------------------------------------------------
if nargin < 3 || isempty(max_iter), max_iter = 100; end
if nargin < 4 || isempty(n_wolves), n_wolves = 30;  end

%% -------------------------------------------------------------------------
%  1. Hyperparameter Bounds
%     [k,   dist_encoded]
%     dist: 0=euclidean, 1=manhattan (continuous encoding, threshold at 0.5)
% -------------------------------------------------------------------------
lb  = [1,  0];
ub  = [20, 1];
dim = 2;

fprintf('AGWO-LS KNN | k in [%d,%d], distance: Euclidean or Manhattan\n',...
    lb(1), ub(1));

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
    fitness(i) = knnObjective(X(i,:), X_train, y_train);
end

[fitness, idx] = sort(fitness);
X = X(idx, :);

X_alpha = X(1,:);  f_alpha = fitness(1);
X_beta  = X(2,:);
X_delta = X(3,:);

conv_curve = zeros(max_iter, 1);

fprintf('\n--- AGWO-LS KNN Optimization Started ---\n');
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
        fitness(i) = knnObjective(X(i,:), X_train, y_train);
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
            f_cand = knnObjective(X_cand, X_train, y_train);
            if f_cand < ls_fitness
                ls_current = X_cand; ls_fitness = f_cand;
                improved = true; break;
            end

            X_cand = ls_current;
            X_cand(j) = max(ls_current(j) - step_j, lb(j));
            f_cand = knnObjective(X_cand, X_train, y_train);
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
    dist_str = decode_distance(X_alpha(2));
    fprintf('Iter %3d/%d | Best CV error = %.4f | k=%d, dist=%s\n',...
        t, max_iter, f_alpha, round(X_alpha(1)), dist_str);
end

%% -------------------------------------------------------------------------
%  5. Output
% -------------------------------------------------------------------------
best_params  = X_alpha;
best_fitness = f_alpha;

fprintf('\n--- Optimization Complete ---\n');
fprintf('Optimal k         = %d\n',   round(best_params(1)));
fprintf('Optimal distance  = %s\n',   decode_distance(best_params(2)));
fprintf('Best CV error     = %.4f\n\n', best_fitness);

end % function AGWO_LS_KNN


%% =========================================================================
%  Local Function: knnObjective
% =========================================================================
function cv_error = knnObjective(params, X_train, y_train)

k         = max(1, round(params(1)));
dist_name = decode_distance(params(2));

try
    mdl      = fitcknn(X_train, y_train, ...
                       'NumNeighbors', k, ...
                       'Distance',     dist_name);
    cv_mdl   = crossval(mdl, 'KFold', 5);
    cv_error = kfoldLoss(cv_mdl);
catch
    cv_error = 1;
end

end % function knnObjective


%% =========================================================================
%  Local Function: decode_distance
% =========================================================================
function dist_name = decode_distance(val)
% Decodes continuous [0,1] encoding to distance metric string
if val < 0.5
    dist_name = 'euclidean';
else
    dist_name = 'cityblock';   % Manhattan distance in MATLAB
end
end
