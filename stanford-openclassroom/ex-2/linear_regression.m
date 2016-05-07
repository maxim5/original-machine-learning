# Solution for
# http://openclassroom.stanford.edu/MainFolder/DocumentPage.php?course=MachineLearning&doc=exercises/ex2/ex2.html

# It's a good practice is to run `clear` first.
clear

x = load('ex2x.dat');
y = load('ex2y.dat');

figure  % Opens a new figure window
plot(x, y, 'o');
ylabel('Height in meters');
xlabel('Age in years');

m = length(y);        % The number of training examples
x = [ones(m, 1), x];  % Add a column of ones to x

# Gradient descent
alpha = 0.07;
theta = [0 0];
counter = 0;
eps = 1e-7;
while (1)
  # Dot product (over i=1..m, j=1..2) of two m-size vectors
  #   dot(theta, x(i,:)) - y(i)
  #   x(i, j)
  # Which is the same as matrix multiply
  dJ = (x * theta' - y)' * x;
  step = alpha / m * dJ;
  theta = theta - step;
  counter = counter + 1;
  
  # norm(step) is too close to zero
  if (abs(step(1)) < eps && abs(step(2)) < eps)
    break
  endif
endwhile

# Learning results and prediction
counter
theta
predict = [1 3.5] * theta'
predict = [1 7] * theta'

hold on # Plot new data without clearing old plot
plot(x(:,2), x * theta', '-')  # remember that x is now a matrix with 2 columns
                               # and the second column contains the time info
legend('Training data', 'Linear regression')


# Understanding $J(\theta)$

J_vals = zeros(100, 100);   % initialize Jvals to 100x100 matrix of 0's
theta0_vals = linspace(-3, 3, 100);
theta1_vals = linspace(-1, 1, 100);
for i = 1:length(theta0_vals)
	for j = 1:length(theta1_vals)
	  t = [theta0_vals(i); theta1_vals(j)];
    v = x * t - y;
	  J_vals(i, j) = 1 / (2*m) * v' * v;
  end
end


% Plot the surface plot
% Because of the way meshgrids work in the surf command, we need to 
% transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals';
figure;
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0'); ylabel('\theta_1')

figure;
% Plot the cost function with 15 contours spaced logarithmically
% between 0.01 and 100
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 2, 15))
xlabel('\theta_0'); ylabel('\theta_1')
