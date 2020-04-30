function plotData02(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx3 matrix.

% Create New Figure
figure; hold on;

% Find Indices of Positive and Negative Examples
pos = find(y==1);
neg = find(y==0);

% Plot Examples
plot(X(pos, 2), X(pos, 3), 'g+', 'LineWidth', 2, 'MarkerSize', 15);
plot(X(neg, 2), X(neg, 3), 'r*', 'MarkerSize', 7);
legend('2.5% or greater drop', 'All other scenarios');
title('Classification visualization');
#mesh(X);

hold off;

end
