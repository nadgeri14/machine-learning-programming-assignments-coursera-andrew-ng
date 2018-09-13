function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
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

h = sigmoid(X*theta);
y_1 = -1 * (y' * log(h));
y_2 = -1 * ( (1 - y)' * log(1 - h) );
sum1 = y_1 + y_2;
sum1 = sum1/m;
regul = sum(theta(2:end).^2);
regul = (lambda / (2*m)) * regul;
J = sum1 + regul;

diff = h - y;
temp = X' * diff;
normal_grad = temp./m;
regul_grad = ( (lambda/m).*(theta(2:end)) );
grad(2:end) = regul_grad;
grad = grad + normal_grad; 



% =============================================================

end
