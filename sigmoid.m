function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

#Acquire dimensions of argument passed to the function
[m,n]=size(z);

#Succesive selection constructs for the single value, vector, and matrix cases
##if (m==1 && n==1) 
##  #disp('Single val')
##  g(1,1)=1/(1+exp(-1*z));
##elseif (m==1 && n>1)
##  #disp('Row vec')
##  for (j=1:1:n)
##    g(1,j)=1/(1+exp(-1*z(1,j)));
##  endfor
##elseif (m>1 && n==1)
##  #disp('Column vec')
##  for (j=1:1:m)
##    g(j,1)=1/(1+exp(-1*z(j,1)));
##  endfor
##elseif(m>1 && n>1)
##  #disp("Matrix")
##  for (j=1:1:m)
##    for (k=1:1:n)
##      g(j,k)=1/(1+exp(-1*z(j,k)));
##    endfor
##  endfor
##else
##  disp('CATASTROPHIC ERROR')
##endif

g=1./(1+exp(-1.*z));

% =============================================================

end
