close all

 t = linspace(0,2*pi,1000);
 theta0 = pi/3;
 a=4;
 b=1;
 x = a*sin(t+theta0) + 4;
 y = b*cos(t) + 5;
 

figure
scatter(x(:), y(:))
xlim([0,10])
ylim([0,10])
title('ellipse data');
xlabel('x');
ylabel('y');

%add noise to the data
x1 = x(:) + rand([numel(x(:)) 1]);
y1 = y(:) + rand([numel(y(:)) 1]);
figure
scatter(x1, y1)
xlim([0,10])
ylim([0,10])
title('noisy ellipse data');
xlabel('x');
ylabel('y');

%surf(x, y, z) 

%collect as data
data = [x1, y1];

%do pca
%1. find mean
mean1 = mean(data);
%2. subtract mean
data_mean = data - repmat(mean1, [numel(x1), 1]);

figure
scatter(data_mean(:,1), data_mean(:,2))
xlim([-5,5])
ylim([-5,5])
title('zero mean noisy ellipse data');
xlabel('x');
ylabel('y');

%3. construct covariance matrix
covMatrix = cov(data_mean);

% find the (diagonal matrix of) eigenvalues
% and eigenvectors of the covariance matrix
[eigenvectors, diagonal] = eig(covMatrix);
% the columns are the eigenvectors

%visualize the eigenvectors and eigenvalues
figure
scatter(data_mean(:,1), data_mean(:,2))
hold on
xlim([-5,5])
ylim([-5,5])
title('zero mean noisy ellipse data with eigenvectors and eigenvalues');
xlabel('x');
ylabel('y');
point1 = sqrt(diagonal(1,1)) * eigenvectors(:, 1);
point2 = sqrt(diagonal(2,2)) * eigenvectors(:, 2);
line([0, point1(1)], [0, point1(2)], 'color', 'r');
line([0, point2(1)], [0, point2(2)], 'color', 'r');

%take the maximum two eigenvalues' indexes as principal components and 
% their eigenvectors 
[~, maxIndexes] = sort(diag(diagonal), 'descend');
pc1Index = maxIndexes(1);
eigenvector1 = eigenvectors(:, 1);

%project the inputs into the space spanned by the principal components
projectionMatrix = [eigenvector1'];
newData = projectionMatrix * data_mean';

%display projected inputs
figure
scatter(newData(1,:), ones(size(newData)))
title('projected noisy ellipse data');
xlabel('x');
ylabel('y');
xlim([-2,2]);
ylim([-2,2]);