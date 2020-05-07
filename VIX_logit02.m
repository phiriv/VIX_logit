#Filename:      VIX_logit02.m
#Author:        Philippe C. Rivet
#Date:          20/04/27
#Description:   Econometrics project to predict stock market drops using  
#               the VIX and its variants through simple logistic regression.
#               In this second script, variants such as the VIX 9D, 3M, and 6M
#               are used as additional features for more 'precise' classification.
#               Thanks to Andrew Ng and the Coursera ML team for inspiration
#               --> coursera.org/learn/machine-learning

#Initialization
clear; close all; clc;

#Load relevant data
vixTS=dlmread('vix.txt');
vix=vixTS(:,2);
vvixTS=dlmread('vvixtimeseries.txt');
vvix=vvixTS(1009:end,2);#3288 --> 2280
vix9dTS=dlmread('vix9ddailyprices.txt');
vix9d=vix9dTS(:,2);#2280
vix3mTS=dlmread('vix3mdailyprices.txt');
vix3m=vix3mTS(777:end,2);#3056 --> 2280
vix6mTS=dlmread('vix6mdailyprices.txt');
vix6m=vix6mTS(755:end,2);#3034 --> 2280
vix1yTS=dlmread('VIX1Y_Data.txt');
vix1y=vix1yTS(1016:end,2);#3295 --> 2280
sptsxTS=dlmread('SPTSX_Data.txt');
#sptsx=sptsxTS(196:end,4:5);
sptsx=sptsxTS(6253:end,4:5);#8532 --> 2280

#5 features (VIX9D, VIX3M, VIX6M, VIX1Y, VVIX) will now be used for LR
#For this reason, using PCA (Principal Component Analysis) is necessary to 
#reduce the dimensionality for visualization purposes

#Drops in the S&P/TSX >= 2.5% are what we're interested in predicting with LR,
#so they are labelled 1 and everything else 0
y=sptsx(:,2);
X=[vvix vix9d vix3m vix6m vix1y];
fprintf('\nData loaded.\n');

#Feature normalizing will speed up computation later on
[X_norm, mu, sigma] = featureNormalize(X);

figure(1);
plot(X_norm);
title('Selected VIX variants');
legend('VVIX', 'VIX9D', 'VIX3M','VIX6M', 'VIX1Y');
title('Normalized features');

#Execute PCA via SVD (Singular Value Decomposition)
[U, S]=pca(X_norm);

fprintf('\nPCA complete, now performing dimensionality reduction.\n\n');
#Project the data onto K = 3 dimensions for demonstration purposes
K = 3;
Z = projectData(X_norm, U, K);
fprintf('Projection of the first example: %f\n', Z(1));
#hold on;
#figure(2);
#plot(Z);

figure(3);
surf(Z);
title('Surface mesh of the features (K=3)');

#Project onto 2 dims for simplicity
Z = projectData(X_norm, U, 2);
figure();
plot(Z);
title('Plot of the further reduced features (K=2)');

fprintf('\nProgram paused. Press any key to continue.\n');
kbhit;

#Now we're ready to perform logistic regression using gradient descent

#Set up the matrix and add a column of unit features
[m, n] = size(Z);
X=[ones(m,1) Z];

#initialize the parameters to be fitted
init_theta=zeros(n+1, 1);

#compute & display initial cost + grad
[cost, grad]=costFunction02(init_theta, X, y);

fprintf('Cost at initial theta (zeros): %f\n', cost);
fprintf('Gradient at initial theta (zeros): \n');
fprintf(' %f \n', grad);

#Compute and display cost and gradient with non-zero test theta
test_theta = [1; -1; 1;];
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

#Display the 'landscape' of the reduced features and events of interest
#plotData02(X, y);
#plotDecisionBoundary02(theta, X, y);

#The decision boundary is non-linear so more features are needed
X2=mapFeature(X(:,2), X(:,3));

#test with theta of all ones and lambda of 100
test_theta = ones(size(X2,2),1);
[cost, grad] = costFunctionReg(test_theta, X2, y, 100);

fprintf('\nCost at test theta (with lambda = 10): %f\n', cost);
fprintf('Gradient at test theta - first five values only:\n');
fprintf(' %f \n', grad(1:5));

fprintf('\nProgram paused. Press any key to continue.\n');
kbhit;

#Finally, play with the lambda value to get a nice boundary with optimal theta
lambda=50;

#similar setup for unconstrained optimization as before
options = optimset('GradObj', 'on', 'MaxIter', 500);

fprintf('\nNow executing regularized logistic regression...\n')

#Bada-bing, bada-boom
[theta, J, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, X2, y, lambda)), test_theta, options);
  
plotDecisionBoundary02(theta, X2, y, lambda);