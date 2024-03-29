function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
% theta is 2 by 1 matrix or 1 column vector, X is m by 2 matrix => hypo
% = m by 1 matrix or vector will be result and since y = a column vector
% it is possible to hypo - y
hypo = X * theta; % hypothesis function (Theta' * x = theta_0 * 1 + theta_1 * x_1) for treating theta_0 as another feature 
square_errors = (hypo - y) .^ 2;%.^2 to apply ^2 to all the elements of column vector.
J = (1/(2*m)) * sum(square_errors);



% =========================================================================

end
