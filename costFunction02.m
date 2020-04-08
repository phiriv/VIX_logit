function [J, grad] = costFunction02(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
% Note: grad should have the same dimensions as theta
%

##for j=1:m
##  subtotal+=( -y(j)*log(sigmoid(theta'*X(j))) - (1-y(j))*(log(1-sigmoid(theta'*X(j)))) );
##  #disp(subtotal)
##end

#Compute vectorized cost using matrix multiplication
eps=1e-10;
##bug1=X*theta;
##bug2=1-sigmoid(bug1+eps);
##bug3=-y'*(bug2+eps);
#bug4=
J=(1/m)*(-y'*log(sigmoid(X*theta) + eps) - (1-y')*(log(1-sigmoid(X*theta)+eps)) );
  
#Similar process for the gradient
hyp=zeros(m,1);
hyp=sigmoid(X*theta);
errors=hyp-y;
grad=(1/m)*(X'*errors);

#theta-=grad;


% =============================================================

end
