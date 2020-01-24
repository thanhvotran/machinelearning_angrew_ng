function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples, y is result we have in training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    hypo = X * theta ; % hypothesis, alpha is learning rate, X contains training examples 1, 
    % m by 2 and 2 by 1 => x by 1 = 1 column vector, ex: x1*theta0
    % x2*theta1
    theta = theta - (alpha * (1 / m) * (X' * (hypo - y)));%column vector * column vector
    % = column vector m by 1 => X' = 2 by m ( X m by 2),
    % NOTE for the sum, considering the matrix multiplication rather than
    % normal sum, each elements of each row of first matrix * each elmts of
    % each column.





    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
