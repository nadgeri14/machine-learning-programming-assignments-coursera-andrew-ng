function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

predict = sigmoid(X * theta);
y1 = -1* (y' * log(predict));
y2 = -1* ( (1 - y)' * log(1 - predict) );
unregularized_sum = y1 + y2;
unregularized_term = unregularized_sum/m;

theta_square = theta([2:end],:).^2;
theta_square_sum = sum(theta_square(:));
regularized_term = (theta_square_sum * lambda)/(2*m);

J = regularized_term + unregularized_term;


diff = predict - y;
temp = X' * diff;
unregularized_gradient = temp./m;
term = lambda/m;
grad([2:end],:) = grad([2:end],:) + term * theta([2:end],:); 
grad = grad + unregularized_gradient;







% =============================================================

grad = grad(:);

end
