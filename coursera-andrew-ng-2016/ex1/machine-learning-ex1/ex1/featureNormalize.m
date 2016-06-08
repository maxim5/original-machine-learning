function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       

for i = 1:size(X, 2)
  column = X(:, i);
  mu(i) = mean(column);
  sigma(i) = std(column);
  X_norm(:,i) = (column - mu(i)) / sigma(i);
end

#house_size = X(:, 1);
#bedrooms = X(:, 2);
#mu = mean(X);
#sigma = std(X);
#house_size_norm = (house_size - mu(1)) / sigma(1);
#bedrooms_norm = (bedrooms - mu(2)) / sigma(2);
#X_norm = [ house_size_norm bedrooms_norm ];

% ============================================================

end
