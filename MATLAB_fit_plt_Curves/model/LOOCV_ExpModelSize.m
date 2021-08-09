% use dataset size as x instead.
% referenced from LOOCV_ExpModel.m


%% parameters:
n_iter_fit = 20; % num of iterations for each a,b,c estimate
n_case = 6; % num of GAN curves
n_test = 1; % num of testing pairs
% an array of all the dataset ID values:
x_tmp = [1000, 2000, 4000, 6000, 8189,...
         200, 600, 1000, 4000, 8000,...
         10000, 30000, 60000,...
         200, 1000, 5000, 10000, 30000];
x_mean_common = mean(x_tmp);
x_std_common = std(x_tmp);


%% (1) biggan FLOWER:
x1 = [1000, 2000, 4000, 6000, 8189]'; % size
x1_norm = (x1-x_mean_common) / x_std_common;
y1 = [76.07, 34.77, 1.86, 4.49, 0.29].'; % rep percent
xx1 = linspace(1000,8500,1000); % for x = size
fitoption1 = fitoptions('Normal', 'off', ...
                        'Method', 'NonlinearLeastSquares', ...
                        'MaxFunEvals', 10000, ...
                        'MaxIter', 10000, ...
                        'TolFun', 1e-10, ...
                        'Lower', [0.00001 500 0], ...
                        'Upper', [0.99999 1000 50], ...
                        'StartPoint', [0.9589, 800, 10]); 
                    
%% (2) stylegan2 FLOWER:
x2 = [1000, 4000, 8189]'; % size
x2_norm = (x2-x_mean_common) / x_std_common;
y2 = [31.93, 1.76, 1.07]';
xx2 = linspace(1000,8500,1000); % for x = size
fitoption2 = fitoptions('Normal', 'off', ...
                        'Method', 'NonlinearLeastSquares', ...
                        'MaxFunEvals', 10000, ...
                        'MaxIter', 10000, ...
                        'TolFun', 1e-10, ...
                        'Lower', [0.00001 500 0], ...
                        'Upper', [0.99999 1000 50], ...
                        'StartPoint', [0.9728, 800, 10]);

%% (3) biggan CelebA:
x3 = [200, 600, 1000, 4000, 8000]'; % size
x3_norm = (x3-x_mean_common) / x_std_common;
y3 = [84.18, 19.82, 4.30, 11.43, 6.05]';
xx3 = linspace(200,8000,1000); % for x = size
fitoption3 = fitoptions('Normal', 'off', ...
                        'Method', 'NonlinearLeastSquares', ...
                        'MaxFunEvals', 10000, ...
                        'MaxIter', 10000, ...
                        'TolFun', 1e-10, ...
                        'Lower', [0.00001 200 -150], ... 
                        'Upper', [0.99999 500 10], ...
                        'StartPoint', [0.9744 300 -50]);

%% (4) stylegan2 CelebA:
x4 = [1000, 4000, 8000]'; % size
x4_norm = (x4-x_mean_common) / x_std_common;
y4 = [76.86, 21.19, 15.14]';
xx4 = linspace(1000,8000,1000); % for x = size
fitoption4 = fitoptions('Normal', 'off', ...
                        'Method', 'NonlinearLeastSquares', ...
                        'MaxFunEvals', 10000, ...
                        'MaxIter', 10000, ...
                        'TolFun', 1e-10, ...
                        'Lower', [0.00001 200 10], ...
                        'Upper', [0.99999 800 110], ...
                        'StartPoint', [0.9671 300 100]);

%% (5) biggan LSUN:
x5 = [200, 1000, 5000, 10000]'; % size
x5_norm = (x5-x_mean_common) / x_std_common;
y5 = [38.57, 27.15, 0, 0.20]';
xx5 = linspace(200,10000,1000); % for x = size
fitoption5 = fitoptions('Normal', 'off', ...
                        'Method', 'NonlinearLeastSquares', ...
                        'MaxFunEvals', 10000, ...
                        'MaxIter', 10000, ...
                        'TolFun', 1e-10, ...
                        'Lower', [0.00001 500 0], ...
                        'Upper', [0.99999 1000 50], ...
                        'StartPoint', [0.9757 800 10]);

%% (6) stylegan2 LSUN:
x6 = [200, 1000, 5000, 10000, 30000]'; % size
x6_norm = (x6-x_mean_common) / x_std_common;
y6 = [92.38, 27.93, 1.17, 3.03, 4.30]';
xx6 = linspace(200,30000,1000); % for x = size
fitoption6 = fitoptions('Normal', 'off', ...
                        'Method', 'NonlinearLeastSquares', ...
                        'MaxFunEvals', 10000, ...
                        'MaxIter', 10000, ...
                        'TolFun', 1e-10, ...
                        'Lower', [0.00001 500 0], ...
                        'Upper', [0.99999 1000 50], ...
                        'StartPoint', [0.9757 800 10]);


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



