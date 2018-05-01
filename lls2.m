%this code demomstrates the use of linear least squares for a line

close all;

%create the data
x = [1 2, 3, 4, 5];
y = [8, 6, 4, 7, 9];

%plot the data
figure;
plot(x, y, '*');
xlim([0,6]);
ylim([0,10]);
xlabel('x');
ylabel('y');
title('data');

%create X and Y matrices
XMatrix = [];
YMatrix = [];
for i = 1 : length(x)
    XMatrix = [XMatrix ; [x(i) * x(i), x(i), 1]];
    YMatrix = [YMatrix ; y(i)];
end

%using linear least squares solution
w =  (inv(XMatrix' * XMatrix))' * XMatrix' * YMatrix

%plot this curve now using w
figure;
plot(x, y, '*');
hold on;
x11 = 1 : 0.1 : 5
y1 = w(1)*x11.*x11 + w(2)*x11 + w(3);
plot(x11, y1);
xlim([0,6]);
ylim([0,10]);
xlabel('x');
ylabel('y');
title('linear least squares curve');