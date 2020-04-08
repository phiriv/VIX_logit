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

#Compute vectorized cost using matrix multiplication as before
eps=1e-10;
J=(1/m)*(-y'*log(sigmoid(X*theta)+eps) - (1-y')*(log(1-sigmoid(X*theta)+eps)) );
#Addition of regularized component
thet1=theta(1);
theta(1)=0;
square=(lambda/(2*m))*(theta'*theta);
#disp(square);
J+=square;
  
#Similar process for the gradient, except now we need to account for different
#cases of theta for the regularized component (e.g. j=0 & j>=1)

theta(1)=thet1;
hyp=zeros(m,1);
hyp=sigmoid(X*theta);
errors=hyp-y;
grad1=(1/m)*(X'*errors);
#disp(grad1);

temp=theta;
temp(1)=0;

grad=0;
#disp(lambda/m);
grad2=temp*(lambda/m);
#disp(grad2);
grad+=grad1+grad2;


% =============================================================

end
