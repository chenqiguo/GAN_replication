% Model the curve of dataset ID (Intrinsic Dimensionality) vs replication percentage:
% the model can predict the replication percentage given a few ID-repPercent pairs (preferabely only one).

% then, we start with an exponential decay function with a translation and scaling on x:
% y=a^(bx-c) with a smaller than 1, x the dataset ID(or size) and y the replication percentage.

% important note: 0<=y<=100 !!!
% note: NO MNIST dataset here.

%{
%% (1) biggan FLOWER:
%x = [22.02, 24.70, 27.41, 28.99, 30.34]'; % ID
x = [1000, 2000, 4000, 6000, 8189]'; % dataset size
y = [76.07, 34.77, 1.86, 4.49, 0.29].'; % rep percent

fitoption = fitoptions('Normal', 'on', ...
                       'Method', 'NonlinearLeastSquares', ...
                       'MaxFunEvals', 10000, ...
                       'MaxIter', 10000, ...
                       'TolFun', 1e-10, ...
                       'Lower', [0.00001 10 0], ...
                       'Upper', [0.99999 130 110]);

g = fittype(@(a, b, c, x) a.^(b*x-c), 'options', fitoption);
f1 = fit(x,y,g);

%xx = linspace(20,32,50); % for x = ID
xx = linspace(500,10000,500); % for x = dataset size
figure;
plot(x,y,'o',xx,f1(xx),'r-');
%}

%% (2) stylegan2 FLOWER:
x = [22.02, 27.41, 30.34]'; % ID
%x = [1000, 4000, 8189]'; % dataset size
y = [31.93, 1.76, 1.07]';

fitoption = fitoptions('Normal', 'off', ...
                       'Method', 'NonlinearLeastSquares', ...
                       'MaxFunEvals', 10000, ...
                       'MaxIter', 10000, ...
                       'TolFun', 1e-10, ...
                       'Lower', [0.00001 10 1000], ...
                       'Upper', [0.99999 130 2000], ...
                        'StartPoint', [0.989, 196.6, 100]);

g = fittype(@(a, b, c, x) a.^(b*x-c), 'options', fitoption);
f2 = fit(x,y,g);

xx = linspace(20,32,50); % for x = ID
%xx = linspace(500,10000,500); % for x = dataset size
figure;
plot(x,y,'o',xx,f2(xx),'r-');

%{
%% (3) biggan CelebA:
%x = [11.90, 15.97, 17.30, 21.34, 23.29]'; % ID
x = [200, 600, 1000, 4000, 8000]'; % dataset size
y = [84.18, 19.82, 4.30, 11.43, 6.05]';

fitoption = fitoptions('Normal', 'on', ...
                       'Method', 'NonlinearLeastSquares', ...
                       'MaxFunEvals', 10000, ...
                       'MaxIter', 10000, ...
                       'TolFun', 1e-10, ...
                       'Lower', [0.00001 10 -100], ...
                       'Upper', [0.99999 130 110]);

g = fittype(@(a, b, c, x) a.^(b*x-c), 'options', fitoption);
f3 = fit(x,y,g);

%xx = linspace(10,25,50); % for x = ID
xx = linspace(100,10000,500); % for x = dataset size
figure;
plot(x,y,'o',xx,f3(xx),'r-');
%}
%{
%% (4) stylegan2 CelebA:
%x = [17.30, 21.34, 23.29]'; % ID
x = [1000, 4000, 8000]'; % dataset size
y = [76.86, 21.19, 15.14]';

fitoption = fitoptions('Normal', 'on', ...
                       'Method', 'NonlinearLeastSquares', ...
                       'MaxFunEvals', 10000, ...
                       'MaxIter', 10000, ...
                       'TolFun', 1e-10, ...
                       'Lower', [0.00001 10 0], ...
                       'Upper', [0.99999 130 100]);

g = fittype(@(a, b, c, x) a.^(b*x-c), 'options', fitoption);
f4 = fit(x,y,g);

%xx = linspace(15,25,50); % for x = ID
xx = linspace(500,10000,500); % for x = dataset size
figure;
plot(x,y,'o',xx,f4(xx),'r-');
%}
%{
%% (5) biggan LSUN:

% to be done after the experiments are finished...
%}
%{
%% (6) stylegan2 LSUN:
x = [14.87, 20.80, 27.06, 29.57, 33.60]'; % ID
%x = [200, 1000, 5000, 10000, 30000]'; % dataset size
y = [92.38, 27.93, 1.17, 3.03, 4.30]';

fitoption = fitoptions('Normal', 'on', ...
                       'Method', 'NonlinearLeastSquares', ...
                       'MaxFunEvals', 10000, ...
                       'MaxIter', 10000, ...
                       'TolFun', 1e-10, ...
                       'Lower', [0.00001 10 -100], ...
                       'Upper', [0.99999 130 110], ...
                       'StartPoint', [0.9775, 70.9, 100]);

g = fittype(@(a, b, c, x) a.^(b*x-c), 'options', fitoption);
f6 = fit(x,y,g);

xx = linspace(10,35,50); % for x = ID
%xx = linspace(100,30000,5000);% for x = dataset size
figure;
plot(x,y,'o',xx,f6(xx),'r-');
%}














%{
%% Toy data:
% Define function that the X values obey.
a = 0.5;
b = 1;
c = 10;
x = linspace(20,32,50)';
y_real = a.^(b*x-c);

% Add noise to y_real.
Y = y_real + 1e-10 * randn(1, length(y_real))';

fitoption = fitoptions('Normal', 'off', ...
                       'Method', 'NonlinearLeastSquares', ...
                       'MaxFunEvals', 10000, ...
                       'MaxIter', 10000, ...
                       'TolFun', 1e-10, ...
                       'Lower', [0.3 0.5 8], ...
                       'Upper', [0.7 1.1 12], ...
                       'StartPoint', [0.4 0.5 9]);

% fit this toy data:
g = fittype(@(a, b, c, x) a.^(b*x-c), 'options', fitoption);
f1 = fit(x,Y,g);
%f1 = fit(x,y_real,g); %,'StartPoint',[0.4,0.8,-9]

figure;
plot(x,Y,'o',x,f1(x),'r-');
%}



