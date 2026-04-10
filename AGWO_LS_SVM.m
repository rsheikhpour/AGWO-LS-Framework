%% =========================================================================
%  AGWO_LS_SVM: Adaptive Grey Wolf Optimizer with Local Search (AGWO-LS)
%                for SVM Hyperparameter Optimization
% =========================================================================
%
%  Reference:
%    Sheikhpour R., Taghipour Zahir S., Pourhosseini F.
%    "Non-Invasive Prediction of Breast Cancer Biomarkers from Blood
%     Inflammatory Indices Using a Hybrid Adaptive Grey Wolf
%     Optimizer-Based Machine Learning Framework"
%    Array, 2026.
%
%  Description:
%    This function optimizes SVM hyperparameters using the AGWO-LS
%    framework for breast cancer biomarker prediction from hematological
%    indices. Supports both RBF and Linear kernel types.
%
%  Inputs:
%    minmaxtrain  : [n_samples x n_features] normalized training data
%    Ctrain       : [n_samples x 1] training labels
%    max_iter     : maximum number of AGWO iterations (default: 100)
%    n_wolves     : population size / number of candidate solutions
%                   (default: 30)
%    kernel_type  : 'rbf' (default) or 'linear'
%
%  Outputs:
%    best_params  : optimized hyperparameters
%                   RBF kernel    -> [C, gamma]
%                   Linear kernel -> [C]
%    best_fitness : best cross-validation error achieved
%    conv_curve   : convergence curve over iterations
%
%  Hyperparameter search ranges (as reported in the paper):
%    RBF kernel    : C in [0.1, 100], gamma in [0.001, 10]
%    Linear kernel : C in [0.1, 100]
%
%  Usage Example:
%    % RBF kernel
%    [params, fitness, curve] = AGWO_LS_SVM(X_train, y_train, 100, 30, 'rbf');
%    C     = params(1);
%    gamma = params(2);
%
%    % Linear kernel
%    [params, fitness, curve] = AGWO_LS_SVM(X_train, y_train, 100, 30, 'linear');
%    C = params(1);
%
% =========================================================================

function [best_params, best_fitness, conv_curve] = AGWO_LS_SVM(...
    minmaxtrain, Ctrain, max_iter, n_wolves, kernel_type)

%% -------------------------------------------------------------------------
%  0. Default Arguments
% -------------------------------------------------------------------------
if nargin < 3 || isempty(max_iter),    max_iter    = 100;   end
if nargin < 4 || isempty(n_wolves),    n_wolves    = 30;    end
if nargin < 5 || isempty(kernel_type), kernel_type = 'rbf'; end

kernel_type = lower(kernel_type);
assert(ismember(kernel_type, {'rbf','linear'}), ...
    'kernel_type must be ''rbf'' or ''linear''.');

%% -------------------------------------------------------------------------
%  1. Hyperparameter Bounds (as reported in the paper)
% -------------------------------------------------------------------------
switch kernel_type
    case 'rbf'
        % [C,     gamma ]
        lb  = [0.1,   0.001];
        ub  = [100,   10   ];
        dim = 2;
        fprintf('Kernel : RBF | C in [%.3f, %.1f], gamma in [%.3f, %.1f]\n',...
            lb(1), ub(1), lb(2), ub(2));

    case 'linear'
        % [C]
        lb  = [0.1];
        ub  = [100 ];
        dim = 1;
        fprintf('Kernel : Linear | C in [%.3f, %.1f]\n', lb(1), ub(1));
end

%% -------------------------------------------------------------------------
%  2. Local Search Parameters (pattern search / coordinate descent)
%     as described in Section 4.3.3 of the paper
% -------------------------------------------------------------------------
ls_step_init  = 0.1;    % initial step size (relative to [0,1] space)
ls_decay      = 0.5;    % step size decay factor on failure
ls_step_min   = 0.001;  % minimum step size (termination threshold)
ls_max_iter   = 10;     % max local search iterations per global iteration

%% -------------------------------------------------------------------------
%  3. Initialize Wolf Population
% -------------------------------------------------------------------------
rng('shuffle');   % ensure different random seeds across runs

X       = rand(n_wolves, dim) .* (ub - lb) + lb;
fitness = inf(n_wolves, 1);

for i = 1:n_wolves
    fitness(i) = svmObjective(X(i,:), minmaxtrain, Ctrain, kernel_type);
end

% Sort by fitness (ascending = minimizing CV error)
[fitness, idx] = sort(fitness);
X = X(idx, :);

% Alpha (best), Beta (2nd), Delta (3rd)
X_alpha = X(1,:);  f_alpha = fitness(1);
X_beta  = X(2,:);  f_beta  = fitness(2);
X_delta = X(3,:);  f_delta = fitness(3);

conv_curve = zeros(max_iter, 1);

fprintf('\n--- AGWO-LS Optimization Started ---\n');
fprintf('Population: %d | Max iterations: %d\n\n', n_wolves, max_iter);

%% -------------------------------------------------------------------------
%  4. Main AGWO-LS Loop
% -------------------------------------------------------------------------
for t = 1:max_iter

    % --- 4.1 Adaptive convergence factor (AGWO strategy) -----------------
    %   Standard GWO uses linear decay: a = 2 - 2*(t/max_iter)
    %   AGWO modulates a based on population fitness diversity
    f_avg = mean(fitness);
    f_min = fitness(1);      % best (sorted)
    f_max = fitness(end);    % worst
    eps   = 1e-10;           % avoid division by zero

    diversity = abs((f_avg - f_min) / (f_max - f_min + eps));
    a = 2 * (1 - t/max_iter) * diversity;

    % --- 4.2 Update Wolf Positions ----------------------------------------
    for i = 1:n_wolves
        for j = 1:dim

            % Contribution from Alpha
            r1 = rand(); r2 = rand();
            A1 = 2*a*r1 - a;
            C1 = 2*r2;
            D_alpha    = abs(C1 * X_alpha(j) - X(i,j));
            X1         = X_alpha(j) - A1 * D_alpha;

            % Contribution from Beta
            r1 = rand(); r2 = rand();
            A2 = 2*a*r1 - a;
            C2 = 2*r2;
            D_beta     = abs(C2 * X_beta(j) - X(i,j));
            X2         = X_beta(j)  - A2 * D_beta;

            % Contribution from Delta
            r1 = rand(); r2 = rand();
            A3 = 2*a*r1 - a;
            C3 = 2*r2;
            D_delta    = abs(C3 * X_delta(j) - X(i,j));
            X3         = X_delta(j) - A3 * D_delta;

            % New position = average of three guiding wolves
            X(i,j) = (X1 + X2 + X3) / 3;
        end

        % Bound enforcement
        X(i,:) = max(X(i,:), lb);
        X(i,:) = min(X(i,:), ub);

        % Fitness evaluation
        fitness(i) = svmObjective(X(i,:), minmaxtrain, Ctrain, kernel_type);
    end

    % Sort population
    [fitness, idx] = sort(fitness);
    X = X(idx, :);

    % Update hierarchy
    X_alpha = X(1,:);  f_alpha = fitness(1);
    X_beta  = X(2,:);  f_beta  = fitness(2);
    X_delta = X(3,:);  f_delta = fitness(3);

    % --- 4.3 Local Search Refinement on Alpha Wolf -----------------------
    %   Pattern search (coordinate descent) with adaptive step size
    %   Applied exclusively to X_alpha at each iteration
    ls_step    = ls_step_init;
    ls_current = X_alpha;
    ls_fitness = f_alpha;

    for ls_iter = 1:ls_max_iter
        improved = false;

        for j = 1:dim
            % Normalize step relative to parameter range
            step_j = ls_step * (ub(j) - lb(j));

            % Perturb dimension j in positive direction
            X_cand      = ls_current;
            X_cand(j)   = min(ls_current(j) + step_j, ub(j));
            f_cand      = svmObjective(X_cand, minmaxtrain, Ctrain, kernel_type);

            if f_cand < ls_fitness
                ls_current = X_cand;
                ls_fitness = f_cand;
                improved   = true;
                break;
            end

            % Perturb dimension j in negative direction
            X_cand      = ls_current;
            X_cand(j)   = max(ls_current(j) - step_j, lb(j));
            f_cand      = svmObjective(X_cand, minmaxtrain, Ctrain, kernel_type);

            if f_cand < ls_fitness
                ls_current = X_cand;
                ls_fitness = f_cand;
                improved   = true;
                break;
            end
        end

        % Step size decay on failure
        if ~improved
            ls_step = ls_step * ls_decay;
        end

        % Termination: step size below threshold
        if ls_step < ls_step_min
            break;
        end
    end

    % Accept local search result if improved
    if ls_fitness < f_alpha
        X_alpha = ls_current;
        f_alpha = ls_fitness;
        X(1,:)  = X_alpha;
        fitness(1) = f_alpha;
    end

    % Record convergence
    conv_curve(t) = f_alpha;

    fprintf('Iter %3d/%d | Best CV error = %.4f | Alpha = %s\n', ...
        t, max_iter, f_alpha, mat2str(round(X_alpha, 4)));
end

%% -------------------------------------------------------------------------
%  5. Output
% -------------------------------------------------------------------------
best_params  = X_alpha;
best_fitness = f_alpha;

fprintf('\n--- Optimization Complete ---\n');
switch kernel_type
    case 'rbf'
        fprintf('Optimal C     = %.4f\n', best_params(1));
        fprintf('Optimal gamma = %.4f\n', best_params(2));
    case 'linear'
        fprintf('Optimal C     = %.4f\n', best_params(1));
end
fprintf('Best CV error = %.4f\n\n', best_fitness);

end % function AGWO_LS_SVM


%% =========================================================================
%  Local Function: svmObjective
%  Evaluates SVM cross-validation error for a given hyperparameter set
% =========================================================================
function cv_error = svmObjective(params, X_train, y_train, kernel_type)
%
%  Uses 5-fold cross-validation on the training set.
%  Fitness = 1 - mean CV accuracy (minimization problem).
%
%  Inputs:
%    params      : hyperparameter vector [C, gamma] for RBF or [C] for Linear
%    X_train     : normalized training features
%    y_train     : training labels
%    kernel_type : 'rbf' or 'linear'

C = params(1);

switch kernel_type
    case 'rbf'
        gamma = params(2);
        t = templateSVM('KernelFunction', 'rbf', ...
                        'BoxConstraint',  C, ...
                        'KernelScale',    1/sqrt(2*gamma));
    case 'linear'
        t = templateSVM('KernelFunction', 'linear', ...
                        'BoxConstraint',  C);
end

try
    cv_model  = crossval(fitcecoc(X_train, y_train, 'Learners', t), ...
                         'KFold', 5);
    cv_error  = kfoldLoss(cv_model);
catch
    cv_error  = 1;   % penalize invalid configurations
end

end % function svmObjective
