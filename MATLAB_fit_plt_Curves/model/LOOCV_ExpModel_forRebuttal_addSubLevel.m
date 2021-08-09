% ONLY for additional subset levels (for rebuttal):

% combined with the old data, fit NEW curves with across dataset validation,
% for threshold = 7000, 8000, 9000, 10000:
% (for each single curve,) Given one pair of dataset id and replication percentage,
% can we predict the replication percentage at an query dataset ID

% Here we use LOOCV: each time use one dataset for validation and use others
% for training (getting a common a and c by average), and then estimate b
% using the testing pair.

% referenced from LOOCV_ExpModel.m

%% parameters:
n_iter_fit = 20; % num of iterations for each a,b,c estimate
n_case = 2; %6; % num of GAN curves
n_test = 1; % num of testing pairs
% an array of all the dataset ID values:
x_tmp = [22.02, 24.70, 27.41, 28.99, 30.34,...
         11.90, 15.97, 17.30, 21.34, 23.29,...
         14.87, 20.80, 27.06, 29.57, 33.60];
x_mean_common = mean(x_tmp);
x_std_common = std(x_tmp);

%% (4) stylegan2 CelebA:
x4 = [15.97, 17.30, 21.34, 23.29]'; % 11.90, ID
x4_norm = (x4-x_mean_common) / x_std_common;
%y4 = [91.02, 76.86, 21.19, 15.14]'; % rep percent thresh 8000
%y4 = [96.29, 91.41, 44.43, 33.01]'; % rep percent thresh 9000
y4 = [98.83, 98.05, 67.77, 54.39]'; % rep percent thresh 10000
%y4 = [79.00, 54.00, 4.59, 3.91]'; % 95.51, rep percent thresh 7000
xx4 = linspace(10,25,50); % for x = ID
fitoption4 = fitoptions('Normal', 'off', ...
                        'Method', 'NonlinearLeastSquares', ...
                        'MaxFunEvals', 10000, ...
                        'MaxIter', 10000, ...
                        'TolFun', 1e-10, ...
                        'Lower', [0.00001 15 90], ...
                        'Upper', [0.99999 500 101], ... %130
                        'StartPoint', [0.9671 56.4762 100]);

%% (5) biggan LSUN:
x5 = [14.87, 20.80, 27.06, 29.57, 33.60]'; % ID
x5_norm = (x5-x_mean_common) / x_std_common;
%y5 = [38.57, 27.15, 0, 0.20, 0.20]'; % rep percent thresh 8000
%y5 = [66.41, 56.54, 0.39, 1.37, 1.76]'; % rep percent thresh 9000
y5 = [88.57, 76.17, 4.10, 8.50, 6.15]'; % rep percent thresh 10000
%y5 = [18.95, 8.79, 0, 0, 6.15]'; % rep percent thresh 7000
xx5 = linspace(10,35,50); % for x = ID
fitoption5 = fitoptions('Normal', 'off', ...
                        'Method', 'NonlinearLeastSquares', ...
                        'MaxFunEvals', 10000, ...
                        'MaxIter', 10000, ...
                        'TolFun', 1e-10, ...
                        'Lower', [0.00001 20 100], ...
                        'Upper', [0.99999 130 102], ...
                        'StartPoint', [0.9757 56.2404 100]);

%% LOOCV:
x_all = {x4,x5};
x_norm_all = {x4_norm,x5_norm};
y_all = {y4,y5};
xx_all = {xx4,xx5};
fitoption_all = {fitoption4,fitoption5};


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

% Then plot curves:
title_list = {'stylegan2 CelebA', 'biggan LSUN'};

for i = 1:n_case % for each GAN curve
    this_title = title_list{i};
    x = x_all{i};
    y = y_all{i};
    xx = xx_all{i};
    % for LOOCV:
    %a = a_common_all(i);
    %b = b_all_real(i);
    %c = c_common_all(i);
    % for fit:
    a = a_all(i);
    b = b_all_fake(i);
    c = c_all(i);
    
    xx_norm = (xx-x_mean_common)/x_std_common;
    y_fit = a.^(b*xx_norm-c);
    
    fi=1;
    for yi = 1:length(y_fit)
        if y_fit(yi) <= 100
            y_fit_(fi) = y_fit(yi);
            xx_(fi) = xx(yi);
            fi = fi+1;
        end
    end
    y_fit = y_fit_;
    xx = xx_;
    
    figure;
    plot(x,y,'.','MarkerSize',60);
    hold on
    plot(xx,y_fit,'r-','LineWidth', 3);
    grid on;
    title(this_title,'FontSize', 20);
    xlabel('ID', 'FontSize', 18);
    ylabel('replication percent', 'FontSize', 18');
    yticks(0 : 20 : 100);
    %xticks(floor(min(xx)) : 5 : ceil(max(xx)));
    xx_tmp = xx_all{i};
    xticks(min(xx_tmp) : 5 : max(xx_tmp)); % 5 for x = ID
    % Get handle to current axes.
    ax = gca;
    % Set x and y font sizes.
    ax.XAxis.FontSize = 18;
    ax.YAxis.FontSize = 18;
    
    clearvars y_fit_ xx_
    
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









