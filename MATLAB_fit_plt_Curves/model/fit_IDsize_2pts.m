% fit between ID and dataset size using only first two points.
% x = ID
% y = dataset size

% referenced from fit_IDsize.m

% NOT use this: results are not good

%% parameters:
%n_degree = 2;


%% (1) FLOWER:
x1 = [22.02, 24.70, 27.41, 28.99, 30.34]'; % ID
xx1 = linspace(15,35,50);
y1 = [1000, 2000, 4000, 6000, 8189]'; % size

x1_ = [22.02, 24.70]'; % ID
y1_ = [1000, 2000]'; % size

f1_ = fit(x1_,y1_,'exp1');

figure
plot(x1,y1,'.','MarkerSize',60);
hold on
plot(xx1,f1_(xx1),'r-','LineWidth', 3);
grid on;
title('FLOWER','FontSize', 20);
xlabel('ID', 'FontSize', 18);
ylabel('dataset size', 'FontSize', 18');
ax = gca;
% Set x and y font sizes.
ax.XAxis.FontSize = 18;
ax.YAxis.FontSize = 18;

correlation_coeff1_ = corr2(y1,f1_(x1));
r_sqr1_ = power(correlation_coeff1_,2);

%% (2) CelebA:
x2 = [11.90, 15.97, 17.30, 21.34, 23.29]'; % ID
xx2 = linspace(10,25,50);
y2 = [200, 600, 1000, 4000, 8000]'; % size

x2_ = [11.90, 15.97]'; % ID
y2_ = [200, 600]'; % size

f2_ = fit(x2_,y2_,'exp1');

figure
plot(x2,y2,'.','MarkerSize',60);
hold on
plot(xx2,f2_(xx2),'r-','LineWidth', 3);
grid on;
title('CelebA','FontSize', 20);
xlabel('ID', 'FontSize', 18);
ylabel('dataset size', 'FontSize', 18');
ax = gca;
% Set x and y font sizes.
ax.XAxis.FontSize = 18;
ax.YAxis.FontSize = 18;

correlation_coeff2_ = corr2(y2,f2_(x2));
r_sqr2_ = power(correlation_coeff2_,2);

%% (3) LSUN:
x3 = [14.87, 20.80, 27.06, 29.57, 33.60]'; % ID
xx3 = linspace(10,35,50);
y3 = [200, 1000, 5000, 10000, 30000]'; % size

x3_ = [14.87, 20.80]'; % ID
y3_ = [200, 1000]'; % size

f3_ = fit(x3_,y3_,'exp1');

figure
plot(x3,y3,'.','MarkerSize',60);
hold on
plot(xx3,f3_(xx3),'r-','LineWidth', 3);
grid on;
title('LSUN','FontSize', 20);
xlabel('ID', 'FontSize', 18);
ylabel('dataset size', 'FontSize', 18');
ax = gca;
% Set x and y font sizes.
ax.XAxis.FontSize = 18;
ax.YAxis.FontSize = 18;

correlation_coeff3_ = corr2(y3,f3_(x3));
r_sqr3_ = power(correlation_coeff3_,2);

%% (4) MNIST:
x4 = [11.86, 12.30, 12.59]'; % ID
xx4 = linspace(11.5,13,50);
y4 = [10000, 30000, 60000]'; % size

x4_ = [11.86, 12.30]'; % ID
y4_ = [10000, 30000]'; % size

f4_ = fit(x4_,y4_,'exp1');

figure
plot(x4,y4,'.','MarkerSize',60);
hold on
plot(xx4,f4_(xx4),'r-','LineWidth', 3);
grid on;
title('MNIST','FontSize', 20);
xlabel('ID', 'FontSize', 18);
ylabel('dataset size', 'FontSize', 18');
ax = gca;
% Set x and y font sizes.
ax.XAxis.FontSize = 18;
ax.YAxis.FontSize = 18;

correlation_coeff4_ = corr2(y4,f4_(x4));
r_sqr4_ = power(correlation_coeff4_,2);

