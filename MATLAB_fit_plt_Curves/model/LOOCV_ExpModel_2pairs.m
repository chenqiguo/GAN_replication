% try the across dataset validation:
% (for each single curve,) Given 2 pairs of dataset id and replication percentage,
% can we predict the replication percentage at an query dataset ID

% Here we use LOOCV: each time use one dataset for validation and use others
% for training (getting a common a and c by average), and then estimate b
% using the testing pairs.

% referenced from LOOCV_ExpModel.m


%% parameters:
n_iter_fit = 20; % num of iterations for each a,b,c estimate
n_case = 6; % num of GAN curves
n_test = 2; %3; % num of testing pairs
% an array of all the dataset ID values:
x_tmp = [22.02, 24.70, 27.41, 28.99, 30.34,...
         11.90, 15.97, 17.30, 21.34, 23.29,...
         14.87, 20.80, 27.06, 29.57, 33.60];
x_mean_common = mean(x_tmp);
x_std_common = std(x_tmp);


%% (1) biggan FLOWER:
x1 = [22.02, 24.70, 27.41, 28.99, 30.34]'; % ID
x1_norm = (x1-x_mean_common) / x_std_common;
y1 = [76.07, 34.77, 1.86, 4.49, 0.29].'; % rep percent
xx1 = linspace(15,35,50); % for x = ID
fitoption1 = fitoptions('Normal', 'off', ...
                        'Method', 'NonlinearLeastSquares', ...
                        'MaxFunEvals', 10000, ...
                        'MaxIter', 10000, ...
                        'TolFun', 1e-10, ...
                        'Lower', [0.00001 10 100], ...
                        'Upper', [0.99999 130 110], ...
                        'StartPoint', [0.9589, 57.9394, 100]); % Note: StartPoint achieved from fitExpModel_v2.m

%% (2) stylegan2 FLOWER:
x2 = [22.02, 27.41, 30.34]'; % ID
x2_norm = (x2-x_mean_common) / x_std_common;
y2 = [31.93, 1.76, 1.07]';
xx2 = linspace(20,35,50); % for x = ID
fitoption2 = fitoptions('Normal', 'off', ...
                        'Method', 'NonlinearLeastSquares', ...
                        'MaxFunEvals', 10000, ...
                        'MaxIter', 10000, ...
                        'TolFun', 1e-10, ...
                        'Lower', [0.00001 10 100], ...
                        'Upper', [0.99999 200 110], ...
                        'StartPoint', [0.9728, 118.6142, 100]);

%% (3) biggan CelebA:
x3 = [11.90, 15.97, 17.30, 21.34, 23.29]'; % ID
x3_norm = (x3-x_mean_common) / x_std_common;
y3 = [84.18, 19.82, 4.30, 11.43, 6.05]';
xx3 = linspace(10,25,50); % for x = ID
fitoption3 = fitoptions('Normal', 'off', ...
                        'Method', 'NonlinearLeastSquares', ...
                        'MaxFunEvals', 10000, ...
                        'MaxIter', 10000, ...
                        'TolFun', 1e-10, ...
                        'Lower', [0.00001 10 100], ... %-100
                        'Upper', [0.99999 130 110], ...
                        'StartPoint', [0.9744 92.1567 100]);

%% (4) stylegan2 CelebA:
x4 = [17.30, 21.34, 23.29]'; % ID
x4_norm = (x4-x_mean_common) / x_std_common;
y4 = [76.86, 21.19, 15.14]';
xx4 = linspace(15,25,50); % for x = ID
fitoption4 = fitoptions('Normal', 'off', ...
                        'Method', 'NonlinearLeastSquares', ...
                        'MaxFunEvals', 10000, ...
                        'MaxIter', 10000, ...
                        'TolFun', 1e-10, ...
                        'Lower', [0.00001 10 100], ...
                        'Upper', [0.99999 130 110], ...
                        'StartPoint', [0.9671 56.4762 100]);

%% (5) biggan LSUN:
x5 = [14.87, 20.80, 27.06, 29.57]'; % ID
x5_norm = (x5-x_mean_common) / x_std_common;
y5 = [38.57, 27.15, 0, 0.20]';
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
y6 = [92.38, 27.93, 1.17, 3.03, 4.30]';
xx6 = linspace(10,35,50); % for x = ID
fitoption6 = fitoptions('Normal', 'off', ...
                        'Method', 'NonlinearLeastSquares', ...
                        'MaxFunEvals', 10000, ...
                        'MaxIter', 10000, ...
                        'TolFun', 1e-10, ...
                        'Lower', [0.00001 10 100], ...
                        'Upper', [0.99999 130 110], ...
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
    
    % estimate b using the 2 testing pairs with smallest x:
    x = x_all{i};
    y = y_all{i};
    [xs, index] = sort(x);
    n_test_ = min(n_test,length(x));
    x_slect = x(index(1:n_test_));
    y_slect = y(index(1:n_test_));
    % normalize x:
    x_slect_norm = (x_slect-x_mean_common)/x_std_common;
    
    % estimate b using x_slect and y_slect:
    fitoption_b = fitoptions('Normal', 'off', ...
                             'Method', 'NonlinearLeastSquares', ...
                             'MaxFunEvals', 10000, ...
                             'MaxIter', 10000, ...
                             'TolFun', 1e-10, ...
                             'Lower', 10, ...
                             'Upper', 500, ...
                             'StartPoint', 56.2404);
    g_b = fittype(@(b, x) a_common.^(b*x-c_common), 'options', fitoption_b);
    f_b = fit(x_slect_norm,y_slect,g_b);
    
    coefficientValues = coeffvalues(f_b);
    b_est = coefficientValues(1);
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
    a = a_common_all(i);
    b = b_all_real(i);
    c = c_common_all(i);
    % for fit:
    %a = a_all(i);
    %b = b_all_fake(i);
    %c = c_all(i);
    
    y_fit = a.^(b*x_norm-c);
    correlation_coeff = corr2(y,y_fit);
    r_sqr = power(correlation_coeff,2);
    r_sqr_all(i) = r_sqr;
end


