% fit between ID and dataset size.
% x = ID
% y = dataset size


%% parameters:
%n_degree = 2;


%% (1) FLOWER:
x1 = [22.02, 24.70, 27.41, 28.99, 30.34]'; % ID
xx1 = linspace(15,35,50);
y1 = [1000, 2000, 4000, 6000, 8189]'; % size
%{
p1 = polyfit(x1,y1,n_degree);
y1_fit = polyval(p1,xx1);
figure
plot(x1,y1,'o')
hold on
plot(xx1,y1_fit)
hold off
%}
f1 = fit(x1,y1,'exp1');

figure
plot(x1,y1,'.','MarkerSize',60);
hold on
plot(xx1,f1(xx1),'r-','LineWidth', 3);
grid on;
title('FLOWER','FontSize', 20);
xlabel('ID', 'FontSize', 18);
ylabel('dataset size', 'FontSize', 18');
ax = gca;
% Set x and y font sizes.
ax.XAxis.FontSize = 18;
ax.YAxis.FontSize = 18;

correlation_coeff1 = corr2(y1,f1(x1));
r_sqr1 = power(correlation_coeff1,2);

%% (2) CelebA:
x2 = [11.90, 15.97, 17.30, 21.34, 23.29]'; % ID
xx2 = linspace(10,25,50);
y2 = [200, 600, 1000, 4000, 8000]'; % size
%{
p2 = polyfit(x2,y2,n_degree);
y2_fit = polyval(p2,xx2);
figure
plot(x2,y2,'o')
hold on
plot(xx2,y2_fit)
hold off
%}
f2 = fit(x2,y2,'exp1');

figure
plot(x2,y2,'.','MarkerSize',60);
hold on
plot(xx2,f2(xx2),'r-','LineWidth', 3);
grid on;
title('CelebA','FontSize', 20);
xlabel('ID', 'FontSize', 18);
ylabel('dataset size', 'FontSize', 18');
ax = gca;
% Set x and y font sizes.
ax.XAxis.FontSize = 18;
ax.YAxis.FontSize = 18;

correlation_coeff2 = corr2(y2,f2(x2));
r_sqr2 = power(correlation_coeff2,2);

%% (3) LSUN:
x3 = [14.87, 20.80, 27.06, 29.57, 33.60]'; % ID
xx3 = linspace(10,35,50);
y3 = [200, 1000, 5000, 10000, 30000]'; % size
%{
p3 = polyfit(x3,y3,n_degree);
y3_fit = polyval(p3,xx3);
figure
plot(x3,y3,'o')
hold on
plot(xx3,y3_fit)
hold off
%}
f3 = fit(x3,y3,'exp1');

figure
plot(x3,y3,'.','MarkerSize',60);
hold on
plot(xx3,f3(xx3),'r-','LineWidth', 3);
grid on;
title('LSUN','FontSize', 20);
xlabel('ID', 'FontSize', 18);
ylabel('dataset size', 'FontSize', 18');
ax = gca;
% Set x and y font sizes.
ax.XAxis.FontSize = 18;
ax.YAxis.FontSize = 18;

correlation_coeff3 = corr2(y3,f3(x3));
r_sqr3 = power(correlation_coeff3,2);

%% (4) MNIST:
x4 = [11.86, 12.30, 12.59]'; % ID
xx4 = linspace(11.5,13,50);
y4 = [10000, 30000, 60000]'; % size
%{
p4 = polyfit(x4,y4,n_degree);
y4_fit = polyval(p4,xx4);
figure
plot(x4,y4,'o')
hold on
plot(xx4,y4_fit)
hold off
%}
f4 = fit(x4,y4,'exp1');

figure
plot(x4,y4,'.','MarkerSize',60);
hold on
plot(xx4,f4(xx4),'r-','LineWidth', 3);
grid on;
title('MNIST','FontSize', 20);
xlabel('ID', 'FontSize', 18);
ylabel('dataset size', 'FontSize', 18');
ax = gca;
% Set x and y font sizes.
ax.XAxis.FontSize = 18;
ax.YAxis.FontSize = 18;

correlation_coeff4 = corr2(y4,f4(x4));
r_sqr4 = power(correlation_coeff4,2);

