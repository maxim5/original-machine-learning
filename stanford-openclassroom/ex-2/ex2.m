% Exercise 2 Linear Regression

% Data is roughly based on 2000 CDC growth figures
% for boys
%
% x refers to a boy's age
% y is a boy's height in meters
%

clear all; close all; clc
x = load('ex2x.dat'); y = load('ex2y.dat');

m = length(y); % number of training examples


% Plot the training data
figure; % open a new figure window
plot(x, y, 'o');
ylabel('Height in meters')
xlabel('Age in years')

% Gradient descent
x = [ones(m, 1) x]; % Add a column of ones to x
theta = zeros(size(x(1,:)))'; % initialize fitting parameters
MAX_ITR = 1500;
alpha = 0.07;

for num_iterations = 1:MAX_ITR
    % This is a vectorized version of the 
    % gradient descent update formula
    % It's also fine to use the summation formula from the videos
    
    % Here is the gradient
    grad = (1/m).* x' * ((x * theta) - y);
    
    % Here is the actual update
    theta = theta - alpha .* grad;
    
    % Sequential update: The wrong way to do gradient descent
    % grad1 = (1/m).* x(:,1)' * ((x * theta) - y);
    % theta(1) = theta(1) + alpha*grad1;
    % grad2 = (1/m).* x(:,2)' * ((x * theta) - y);
    % theta(2) = theta(2) + alpha*grad2;
end
% print theta to screen
theta

% Plot the linear fit
hold on; % keep previous plot visible
plot(x(:,2), x*theta, '-')
legend('Training data', 'Linear regression')
hold off % don't overlay any more plots on this figure

% Closed form solution for reference
% You will learn about this method in future videos
exact_theta = (x' * x)\x' * y

% Predict values for age 3.5 and 7
predict1 = [1, 3.5] *theta
predict2 = [1, 7] * theta


% Calculate J matrix

% Grid over which we will calculate J
theta0_vals = linspace(-3, 3, 100);
theta1_vals = linspace(-1, 1, 100);

% initialize J_vals to a matrix of 0's
J_vals = zeros(length(theta0_vals), length(theta1_vals));

for i = 1:length(theta0_vals)
	  for j = 1:length(theta1_vals)
	  t = [theta0_vals(i); theta1_vals(j)];    
	  J_vals(i,j) = (0.5/m) .* (x * t - y)' * (x * t - y);
    end
end

% Because of the way meshgrids work in the surf command, we need to 
% transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals';
% Surface plot
figure;
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0'); ylabel('\theta_1');

% Contour plot
figure;
% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 2, 15))
xlabel('\theta_0'); ylabel('\theta_1');
