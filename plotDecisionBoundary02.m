function plotDecisionBoundary02(theta, X, y, lambda)
%PLOTDECISIONBOUNDARY02 Plots the data points X and y into a new figure with
%the decision boundary defined by theta in 3D
%   PLOTDECISIONBOUNDARY(theta, X,y) plots the data points with + for the 
%   positive examples and o for the negative examples. X is assumed to be 
%   a either 
%   1) Mx3 matrix, where the first column is an all-ones column for the 
%      intercept.
%   2) MxN, N>3 matrix, where the first column is all-ones

% Plot Data
plotData02(X, y);
#axis([1 160 -500 18000]);
hold on;

if size(X, 2) <= 3
    % Only need 2 points to define a line, so choose two endpoints
    plot_x = [min(X(:,2))-2,  max(X(:,2))+2];

    % Calculate the decision boundary line
    plot_y = (-1./theta(3)).*(theta(2).*plot_x + theta(1));

    % Plot, and adjust axes for better viewing
    plot(plot_x, plot_y)
    
    % Legend, specific for the exercise
    legend('2.5% or greater drop', 'All other scenarios','Decision boundary');
    #axis([30, 100, 30, 100])
else
    % Here is the grid range
    u = linspace(-10, 4, 1000);
    v = linspace(-10, 6, 1000);

    z = zeros(length(u), length(v));
    % Evaluate z = theta*x over the grid
    for i = 1:length(u)
        for j = 1:length(v)
            z(i,j) = mapFeature(u(i), v(j))*theta;
        end
    end
    z = z'; % important to transpose z before calling contour

    % Plot z = 0
    % Notice you need to specify the range [0, 0]
    contour(u, v, z, 1, 'LineWidth', 2);
    legend('2.5% or greater drop', 'All other scenarios','Decision boundary');
    axis([-10 4 -5 5]);

    X1=X(:,2:3);
    pts=[X1 y];
    idx=(pts==1);
    #find(idx);
    #r=find(idx);
    [r,c]=find(idx);
    r=[r];
    c=[c];#avoid bug where #VAL appears in variable editor
    m=size(r);
    pos_pts=zeros(m,2);
    for j=1:m
      for i=1:2
        pos_pts(j,i)=pts(r(j),i);
      end
    end
    
    k=convhull(pos_pts(:,1),pos_pts(:,2));
    a=pos_pts(:,1);
    b=pos_pts(:,2);
    plot(a(k),b(k));
    title(sprintf('Classification visualization, lambda = %g', lambda));

#https://stackoverflow.com/questions/34052115/how-to-find-the-importance-of-the-features-for-a-logistic-regression-model
    
end
hold off

end
