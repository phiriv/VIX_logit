#Filename:      VIX_logit.m
#Author:        Philippe C. Rivet
#Date:          20/03/30
#Description:   Econometrics project to predict stock market drops using  
#               the VIX and its variants through simple logistic regression.
#               Thanks to Andrew Ng and the Coursera ML course for inspiration
#               --> coursera.org/learn/machine-learning

#Initialization
clear; close all; clc;

#Load relevant data
vixTS=dlmread('vix.txt');
vix=vixTS(:,2);
vvixTS=dlmread('vvixtimeseries.txt');
vvix=vvixTS(:,2);
vix9dTS=dlmread('vix9ddailyprices.txt');
vix9d=vix9dTS(:,2);
vix3mTS=dlmread('vix3mdailyprices.txt');
vix3m=vix3mTS(:,2);
vix6mTS=dlmread('vix6mdailyprices.txt');
vix6m=vix6mTS(:,2);
vix1yTS=dlmread('VIX1Y_Data.txt');
vix1y=vix1yTS(:,2);
sptsxTS=dlmread('SPTSX_Data.txt');
sptsx=sptsxTS(199:end,4:5);
#Drops in the S&P/TSX >= 2.5% are what we're interested in predicting with LR,
#so they are labelled 1 and everything else 0
y=sptsx(:,2);
X=[vix sptsx(:,1)];#initially the only predictors are the VIX and S&P;
                   #more will be added in later for greater than 2 dims

#1st, plot the data to get a bird's eye view
fprintf(['Plotting data with + indicating (y = 1) examples and o ' ...
         'indicating (y = 0) examples.\n']);
         
plotData(X, y);
hold on;
#What are we looking at exactly?
xlabel('VIX');
ylabel('S&P/TSX');
legend('2.5% or greater drop', 'All other scenarios')
hold off;

#Next, we compute the cost function and gradient for logistic regression

#Set up the matrix and add a column of unit features
[m, n] = size(X);
X=[ones(m,1) X];

#initialize the parameters to be fitted
init_theta=zeros(n+1, 1);

#compute & display initial cost + grad
[cost, grad]=costFunction02(init_theta, X, y);

fprintf('Cost at initial theta (zeros): %f\n', cost);
fprintf('Gradient at initial theta (zeros): \n');
fprintf(' %f \n', grad);

#Compute and display cost and gradient with non-zero test theta
test_theta = [-1; 1; 1];
[cost, grad] = costFunction02(test_theta, X, y);

fprintf('\nCost at test theta: %f\n', cost);
fprintf('Gradient at test theta: \n');
fprintf(' %f \n', grad);

fprintf('\nProgram paused. Press any key to continue.\n');
kbhit;

#Now that we've established that the process works, it's time to optimize!

#Set options for function unconstrained optimization
options=optimset('GradObj', 'on', 'MaxIter', 500);
#Run it to obtain the optimal parameters
[theta, cost]= fminunc(@(t)(costFunction02(t,X,y)), init_theta, options);

#Display theta
fprintf('Cost at theta found by fminunc: %f\n', cost);
fprintf('theta: \n');
fprintf(' %f \n', theta);

#plot the predicted boundary
plotDecisionBoundary(theta, X, y);

#Labellios
hold on;
xlabel('VIX')
ylabel('S&PTSX')
legend('>=2.5% drop', 'All other scenarios')
hold off;

#The decision boundary is non-linear so more features are needed
#plotData(X(:,2:3),y);
X=mapFeature(X(:,2), X(:,3));

#initialize params, including the starting lambda for regularization
init_theta=zeros(size(X,2),1);
lambda=1;

[cost, grad]= costFunctionReg(init_theta, X, y, lambda);

fprintf('Cost at initial theta (zeros): %f\n', cost);
fprintf('Gradient at initial theta (zeros) - first five values only:\n');
fprintf(' %f \n', grad(1:5));

#test with theta of all ones and lambda of 100
test_theta = ones(size(X,2),1);
[cost, grad] = costFunctionReg(test_theta, X, y, 100);

fprintf('\nCost at test theta (with lambda = 10): %f\n', cost);
fprintf('Gradient at test theta - first five values only:\n');
fprintf(' %f \n', grad(1:5));

fprintf('\nProgram paused. Press any key to continue.\n');
kbhit;

#Finally, play with the lambda value to get a nice boundary with optimal theta
lambda=50;

#similar setup for unconstrained optimization as before
options = optimset('GradObj', 'on', 'MaxIter', 500);

fprintf('\nNow executing regularized logistic regression...')

#Bada-bing, bada-boom
[theta, J, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, X, y, lambda)), init_theta, options);
  
#Display
#X=[vix sptsx(:,1)];
plotDecisionBoundary(theta, X, y);
hold on;
title(sprintf('lambda = %g', lambda))

xlabel('VIX')
ylabel('S&PTSX')

legend('>=2.5% drop', 'All other scenarios','Decision boundary')

hold off;

fprintf('\nProgram paused. Press any key to continue.\n');
kbhit;