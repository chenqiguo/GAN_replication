% try the across dataset validation:
% (for each single curve,) Given one pair of dataset id and replication percentage,
% can we predict the replication percentage at an query dataset ID

% Here we use LOOCV: each time use one dataset for validation and use others
% for training (getting a common a and c by average), and then estimate b
% using the testing pair.

% referenced from fitExpModel_v2.m


%% parameters:
n_iter_fit = 20; % num of iterations for each a,b,c estimate
n_case = 6; % num of GAN curves
n_test = 1; % num of testing pairs
% an array of all the dataset ID values:
x_tmp = [22.02, 24.70, 27.41, 28.99, 30.34,...
         11.90, 15.97, 17.30, 21.34, 23.29,...
         14.87, 20.80, 27.06, 29.57, 33.60];
x_mean_common = mean(x_tmp);
x_std_common = std(x_tmp);


%% (1) biggan FLOWER:
x1 = [22.02, 24.70, 27.41, 28.99, 30.34]'; % ID
x1_norm = (x1-x_mean_common) / x_std_common;
%y1 = [76.07, 34.77, 1.86, 4.49, 0.29].'; % rep percent thresh 8000
%y1 = [85.55, 53.71, 6.74, 17.97, 1.56].'; % rep percent thresh 9000
%y1 = [89.75, 65.82, 16.89, 36.43, 5.37].'; % rep percent thresh 10000
y1 = [66.11, 18.95, 0.10, 1.17, 0].'; % rep percent thresh 7000
xx1 = linspace(15,35,50); % for x = ID
fitoption1 = fitoptions('Normal', 'off', ...
                        'Method', 'NonlinearLeastSquares', ...
                        'MaxFunEvals', 10000, ...
                        'MaxIter', 10000, ...
                        'TolFun', 1e-10, ...
                        'Lower', [0.00001 10 100], ...
                        'Upper', [0.99999 130 102], ... %110
                        'StartPoint', [0.9589, 57.9394, 100]); % Note: StartPoint achieved from fitExpModel_v2.m

%% (2) stylegan2 FLOWER:
x2 = [22.02, 27.41, 30.34]'; % ID
x2_norm = (x2-x_mean_common) / x_std_common;
%y2 = [31.93, 1.76, 1.07]';  % rep percent thresh 8000
%y2 = [47.66, 7.42, 4.00]';  % rep percent thresh 9000
%y2 = [63.09, 16.31, 10.74]';  % rep percent thresh 10000
y2 = [15.53, 0.29, 0.20]';  % rep percent thresh 7000
xx2 = linspace(20,35,50); % for x = ID
fitoption2 = fitoptions('Normal', 'off', ...
                        'Method', 'NonlinearLeastSquares', ...
                        'MaxFunEvals', 10000, ...
                        'MaxIter', 10000, ...
                        'TolFun', 1e-10, ...
                        'Lower', [0.00001 10 100], ...
                        'Upper', [0.99999 300 102], ... %200
                        'StartPoint', [0.9728, 118.6142, 100]);

%% (3) biggan CelebA:
x3 = [11.90, 15.97, 17.30, 21.34, 23.29]'; % ID
x3_norm = (x3-x_mean_common) / x_std_common;
%y3 = [84.18, 19.82, 4.30, 11.43, 6.05]'; % rep percent thresh 8000
%y3 = [92.09, 35.45, 11.23, 25.29, 20.21]'; % rep percent thresh 9000
%y3 = [96.78, 53.52, 24.51, 43.16, 38.77]'; % rep percent thresh 10000
y3 = [73.34, 8.59, 0.68, 2.34, 0.98]'; % rep percent thresh 7000
xx3 = linspace(10,25,50); % for x = ID
fitoption3 = fitoptions('Normal', 'off', ...
                        'Method', 'NonlinearLeastSquares', ...
                        'MaxFunEvals', 10000, ...
                        'MaxIter', 10000, ...
                        'TolFun', 1e-10, ...
                        'Lower', [0.00001 10 100], ... %-100
                        'Upper', [0.99999 500 102], ... %130
                        'StartPoint', [0.9744 92.1567 100]);

%% (4) stylegan2 CelebA:
x4 = [17.30, 21.34, 23.29]'; % ID
x4_norm = (x4-x_mean_common) / x_std_common;
%y4 = [76.86, 21.19, 15.14]'; % rep percent thresh 8000
%y4 = [91.41, 44.43, 33.01]'; % rep percent thresh 9000
%y4 = [98.05, 67.77, 54.39]'; % rep percent thresh 10000
y4 = [54.00, 4.59, 3.91]'; % rep percent thresh 7000
xx4 = linspace(15,25,50); % for x = ID
fitoption4 = fitoptions('Normal', 'off', ...
                        'Method', 'NonlinearLeastSquares', ...
                        'MaxFunEvals', 10000, ...
                        'MaxIter', 10000, ...
                        'TolFun', 1e-10, ...
                        'Lower', [0.00001 10 100], ...
                        'Upper', [0.99999 500 102], ... %130
                        'StartPoint', [0.9671 56.4762 100]);

%% (5) biggan LSUN:
x5 = [14.87, 20.80, 27.06, 29.57]'; % ID
x5_norm = (x5-x_mean_common) / x_std_common;
%y5 = [38.57, 27.15, 0, 0.20]'; % rep percent thresh 8000
%y5 = [66.41, 56.54, 0.39, 1.37]'; % rep percent thresh 9000
%y5 = [88.57, 76.17, 4.10, 8.50]'; % rep percent thresh 10000
y5 = [18.95, 8.79, 0, 0]'; % rep percent thresh 7000
xx5 = linspace(10,35,50); % for x = ID
fitoption5 = fitoptions('Normal', 'off', ...
                        'Method', 'NonlinearLeastSquares', ...
                        'MaxFunEvals', 10000, ...
                        'MaxIter', 10000, ...
                        'TolFun', 1e-10, ...
                        'Lower', [0.00001 10 100], ...
                        'Upper', [0.99999 130 102], ...
                        'StartPoint', [0.9757 56.2404 100]);

%% (6) stylegan2 LSUN:
x6 = [14.87, 20.80, 27.06, 29.57, 33.60]'; % ID
x6_norm = (x6-x_mean_common) / x_std_common;
%y6 = [92.38, 27.93, 1.17, 3.03, 4.30]'; % rep percent thresh 8000
%y6 = [96.09, 42.19, 4.69, 10.06, 12.01]'; % rep percent thresh 9000
%y6 = [98.14, 61.91, 13.87, 22.56, 22.36]'; % rep percent thresh 10000
y6 = [85.84, 15.53, 0.29, 0.98, 0.88]'; % rep percent thresh 7000
xx6 = linspace(10,35,50); % for x = ID
fitoption6 = fitoptions('Normal', 'off', ...
                        'Method', 'NonlinearLeastSquares', ...
                        'MaxFunEvals', 10000, ...
                        'MaxIter', 10000, ...
                        'TolFun', 1e-10, ...
                        'Lower', [0.00001 10 100], ...
                        'Upper', [0.99999 130 102], ... %110
                        'StartPoint', [0.9757 56.2404 100]);

%% LOOCV:
x_all = {x1,x2,x3,x4,x5,x6};
x_norm_all = {x1_norm,x2_norm,x3_norm,x4_norm,x5_norm,x6_norm};
y_all = {y1,y2,y3,y4,y5,y6};
xx_all = {xx1,xx2,xx3,xx4,xx5,xx6};
fitoption_all = {fitoption1,fitoption2,fitoption3,fitoption4,fitoption5,fitoption6};


% (1) for each curve, estimate its a,(b,)c values
a_all = zeros(1,n_case);
b_all_fake = zeros(1,n_case); % b that estimated from averaging fits --> just for reference!
c_all = zeros(1,n_case);

for i = 1:n_case % for each GAN curve
    x_norm = x_norm_all{i};
    y = y_all{i};
    %xx = xx_all{i};
    fitoption = fitoption_all{i};
    
    a_mat = zeros(1,n_iter_fit);
    b_mat_fake = zeros(1,n_iter_fit);
    c_mat = zeros(1,n_iter_fit);
    
    for j = 1:n_iter_fit % for each estimation iteration
        g = fittype(@(a, b, c, x) a.^(b*x-c), 'options', fitoption);
        f = fit(x_norm,y,g);
        
        coefficientValues = coeffvalues(f);
        a_ = coefficientValues(1);
        b_ = coefficientValues(2);
        c_ = coefficientValues(3);
        
        a_mat(j) = a_;
        b_mat_fake(j) = b_;
        c_mat(j) = c_;
    end
    
    a = mean(a_mat);
    b_fake = mean(b_mat_fake);
    c = mean(c_mat);
    
    a_all(i) = a;
    b_all_fake(i) = b_fake;
    c_all(i) = c;
    
end


% (2) LOOCV:
a_common_all = zeros(1,n_case); % averaged from training --> what we want
b_all_real = zeros(1,n_case); % b that estimated from the testing pair --> what we want
c_common_all = zeros(1,n_case); % averaged from training --> what we want

for i = 1:n_case % for each GAN curve: treated as tesing
    
    % get a and c val by averaging from training:
    a_common = 0;
    c_common = 0;
    for j = 1:n_case
        if i~=j
            a_common = a_common + a_all(j);
            c_common = c_common + c_all(j);
        end
    end
    a_common = a_common / (n_case-1);
    c_common = c_common / (n_case-1);
    a_common_all(i) = a_common;
    c_common_all(i) = c_common;
    
    % estimate b using the testing pair:
    % select the (x,y) with smallest x value:
    x = x_all{i};
    y = y_all{i};
    [xs, index] = sort(x);
    x_slect = x(index(1:n_test));
    y_slect = y(index(1:n_test));
    % normalize x_slect:
    x_slect_norm = (x_slect-x_mean_common)/x_std_common;
    
    logAy = log(y_slect) / log(a_common);
    b_est = (logAy+c_common) / x_slect_norm;
    b_all_real(i) = b_est;
    
end

% (3) calculate the goodness-of-fit (R^2) measurement:
% R-square is the square of the correlation between the response values (i.e. y)
% and the predicted response values (i.e. y_fit)
r_sqr_all = zeros(1,n_case); 
for i = 1:n_case % for each GAN curve
    x_norm = x_norm_all{i};
    y = y_all{i};
    % for LOOCV:
    %a = a_common_all(i);
    %b = b_all_real(i);
    %c = c_common_all(i);
    % for fit:
    a = a_all(i);
    b = b_all_fake(i);
    c = c_all(i);
    
    y_fit = a.^(b*x_norm-c);
    correlation_coeff = corr2(y,y_fit);
    r_sqr = power(correlation_coeff,2);
    r_sqr_all(i) = r_sqr;
end

% (4) compute MAE: just for fit
MAE_fit_all = zeros(1,n_case);
for i = 1:n_case % for each GAN curve
    ID_norm = x_norm_all{i};
    
    rep = y_all{i};
    a_fit_IDrep = a_all(i);
    b_fit_IDrep = b_all_fake(i);
    c_fit_IDrep = c_all(i);
    
    % the id->replication func:
    % (1) full set fit:
    syms x
    f_fit_IDrep = a_fit_IDrep^(b_fit_IDrep*x-c_fit_IDrep);
    MAE_fit_IDrep = computeMAE_func(f_fit_IDrep, ID_norm, rep);
    
    MAE_fit_all(i) = MAE_fit_IDrep;
    
end




