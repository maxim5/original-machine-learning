% Exercise 3: Multivariate Linear Regression

clear all; close all; clc
x = load('ex3x.dat'); y = load('ex3y.dat');

m = length(y);
x = [ones(m, 1), x];
sigma = std(x);
mu = mean(x);
x(:,2) = (x(:,2) - mu(2))./ sigma(2);
x(:,3) = (x(:,3) - mu(3))./ sigma(3);


# Selecting a learning rate
alpha = 0.17;
theta = [0 0 0];
J = zeros(50, 1); 

for num_iterations = 1:50
    tmp = x * theta' - y;
    J(num_iterations) = 1 / (2 * m) .* tmp' * tmp;
    theta = theta - alpha / m .* tmp' * x;
end

% now plot J
% technically, the first J starts at the zero-eth iteration
% but Matlab/Octave doesn't have a zero index
figure;
plot(0:49, J(1:50), '-')
xlabel('Number of iterations')
ylabel('Cost J')


# Gradient descent
theta = [0 0 0];
eps = 1e-7;
counter = 0;
while (1)
  step = alpha / m .* (x * theta' - y)' * x;
  theta = theta - step;
  counter = counter + 1;
  if (abs(step(1)) < eps && abs(step(2)) < eps)
    break
  endif
endwhile

% force Matlab to display more than 4 decimal places
% formatting persists for rest of this session
format long

counter
theta
input = [1 ((1650 - mu(2)) / sigma(2)) ((3 - mu(3)) / sigma(3))];
predict = input * theta'


# Exact solution

theta_exact = inv(x' * x) * x' * y
predict = input * theta_exact
