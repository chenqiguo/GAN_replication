% Model the curve of dataset ID (Intrinsic Dimensionality) vs replication percentage:
% the model can predict the replication percentage given a few ID-repPercent pairs (preferabely only one).

% first, we start with an exponential decay function:
% y=a*exp(b*x) with x the dataset ID(or size) and y the replication percentage.

% important note: 0<=y<=100 !!!
% note: NO MNIST dataset here.

%% (1) biggan FLOWER:
%x = [22.02, 24.70, 27.41, 28.99, 30.34]'; % ID
x = [1000, 2000, 4000, 6000, 8189]'; % dataset size
y = [76.07, 34.77, 1.86, 4.49, 0.29].'; % rep percent

f1 = fit(x,y,'exp1');
% f1 = 3.622e+05*exp(-0.3838*x) for x = ID;
% f1 = 177.7*exp(-0.0008421*x) for x = dataset size

%xx = linspace(20,32,50); % for x = ID
xx = linspace(500,10000,500); % for x = dataset size
figure;
plot(x,y,'o',xx,f1(xx),'r-');

%% (2) stylegan2 FLOWER:
%x = [22.02, 27.41, 30.34]'; % ID
x = [1000, 4000, 8189]'; % dataset size
y = [31.93, 1.76, 1.07]';

f2 = fit(x,y,'exp1');
% f2 = 2.759e+06e+05*exp(-0.5162*x) for x = ID;
% f2 = 83.19*exp(-0.0009576*x) for x = dataset size

%xx = linspace(20,32,50); % for x = ID
xx = linspace(500,10000,500); % for x = dataset size
figure;
plot(x,y,'o',xx,f2(xx),'r-');

%% (3) biggan CelebA:
%x = [11.90, 15.97, 17.30, 21.34, 23.29]'; % ID
x = [200, 600, 1000, 4000, 8000]'; % dataset size
y = [84.18, 19.82, 4.30, 11.43, 6.05]';

f3 = fit(x,y,'exp1');
% f3 = 7467*exp(-0.3771*x) for x = ID;
% f3 = 174.2*exp(-0.003634*x) for x = dataset size

%xx = linspace(10,25,50); % for x = ID
xx = linspace(100,10000,500); % for x = dataset size
figure;
plot(x,y,'o',xx,f3(xx),'r-');

%% (4) stylegan2 CelebA:
%x = [17.30, 21.34, 23.29]'; % ID
x = [1000, 4000, 8000]'; % dataset size
y = [76.86, 21.19, 15.14]';

f4 = fit(x,y,'exp1');
% f4 = 1.319e+04*exp(-0.2975*x) for x = ID;
% f4 = 108.2*exp(-0.0003552*x) for x = dataset size

%xx = linspace(15,25,50); % for x = ID
xx = linspace(500,10000,500); % for x = dataset size
figure;
plot(x,y,'o',xx,f4(xx),'r-');

%% (5) biggan LSUN:

% to be done after the experiments are finished...

%% (6) stylegan2 LSUN:
%x = [14.87, 20.80, 27.06, 29.57, 33.60]'; % ID
x = [200, 1000, 5000, 10000, 30000]'; % dataset size
y = [92.38, 27.93, 1.17, 3.03, 4.30]';

f6 = fit(x,y,'exp1');
% f6 = 2380*exp(-0.2183*x) for x = ID;
% f6 = 124.6*exp(-0.001494*x) for x = dataset size

%xx = linspace(10,35,50); % for x = ID
xx = linspace(100,30000,5000);% for x = dataset size
figure;
plot(x,y,'o',xx,f6(xx),'r-');










