# Exercise 4: Logistic Regression and Newton's Method
# http://openclassroom.stanford.edu/MainFolder/DocumentPage.php?course=MachineLearning&doc=exercises/ex4/ex4.html

clear all; close all; clc
x = load('ex4x.dat'); y = load('ex4y.dat');

m = length(y); x = [ones(m, 1), x];

# Plot the data

% find returns the indices of the
% rows meeting the specified condition
pos = find(y == 1); neg = find(y == 0);

% Assume the features are in the 2nd and 3rd
% columns of x
plot(x(pos, 2), x(pos,3), '+'); hold on
plot(x(neg, 2), x(neg, 3), 'o')


# Newton's Method

% Usage: To find the value of the sigmoid 
% evaluated at 2, call sigmoid(2)
sigmoid = inline('1.0 ./ (1.0 + exp(-z))'); 

MAX_ITR = 10;
theta = zeros(size(x(1,:)));
J = zeros(MAX_ITR, 1); 

for i = 1:MAX_ITR
  h = sigmoid(x * theta');
  dJ = 1 / m .* (h - y)' * x;
  H = 1 / m .* x' * diag(h) * diag(1 - h) * x;
  step = inv(H) * dJ';
  theta = theta - step';
  J(i) = 1 / m .* sum(-y.*log(h) - (1-y).*log(1-h));
end

# Answers
theta
prob = 1 - sigmoid([1, 20, 80] * theta')
J

# Plot the results

% Only need 2 points to define a line, so choose two endpoints
plot_x = [min(x(:,2))-2,  max(x(:,2))+2];
% Calculate the decision boundary line
plot_y = (-1./theta(3)) .* (theta(2).*plot_x + theta(1));
plot(plot_x, plot_y)
legend('Admitted', 'Not admitted', 'Decision Boundary')
hold off

figure
plot(0:MAX_ITR-1, J, 'o--', 'MarkerFaceColor', 'r', 'MarkerSize', 8)
xlabel('Iteration'); ylabel('J')
