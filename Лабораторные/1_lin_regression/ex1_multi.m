%% Initialization

%% ================ Part 6-7: Feature Normalization ================

%% Clear and Close Figures
clear ; close all; clc

fprintf('Loading data ...\n');

%% Load Data
data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');

[X mu sigma] = featureNormalize(X);

% Add intercept term to X
X = [ones(m, 1) X];

%% ================ Part 8: Gradient Descent ================

fprintf('Running gradient descent ...\n');

% Choose some alpha value
alpha = 0.01;
num_iters = 400;

% Init Theta and Run Gradient Descent 
theta = zeros(3, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);
theta_grad = theta;

% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

pause;

%% ================ Part 10: Gradient Descent ================

figure;
fprintf('Choose parameter alpha ...\n');

alpha = 0.01; 
theta = zeros(3, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);
hold on;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2)
  
alpha = 0.02; 
theta = zeros(3, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);  
hold on;
plot(1:numel(J_history), J_history, '-r', 'LineWidth', 2)

alpha = 0.03; 
theta = zeros(3, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);  
hold on;
plot(1:numel(J_history), J_history, '-g', 'LineWidth', 2)

legend('alpha = 0.01', 'alpha = 0.02', 'alpha = 0.03')

pause;

%% ================ Part 11: Normal Equations ================

fprintf('Solving with normal equations...\n');

%% Load Data
data = csvread('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Add intercept term to X
X = [ones(m, 1) X];

% Calculate the parameters from the normal equation
theta = normalEqn(X, y);

% Display normal equation's result
fprintf('Theta computed from the normal equations and gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');

example = [1 1650 3];

example_grad = [1650 3]
example_grad = (example_grad .- mu) ./ sigma;
example_grad = [1, example_grad];
price_normal = example * theta;
price_grad = example_grad * theta_grad;


% ============================================================

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using normal equations):\n $%f\n'], price_normal);

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using grad equations):\n $%f\n'], price_grad);

