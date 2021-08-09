% ONLY for new metric (inceptionv3 comb pixel-wise NN) on FLOWER_128_xxx (for rebuttal):

% plot scatters & fitted curves of ID vs replication,
% for threshold = 33, 36, 39, 42.
% This is for rebuttal.

% NOTE: the new fitting curves were generated in
% LOOCV_ExpModel_forRebuttal_newMetric.m

% referenced from plt_IDvsRep_forSuppMat.m

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

%% (1) biggan FLOWER:
x1 = [22.02, 24.70, 27.41, 28.99, 30.34]'; % ID
x1_norm = (x1-x_mean_common) / x_std_common;
xx1 = linspace(20,35,50); % for x = ID
% rep percent thresh 36:
y1_8000 = [87.70, 38.67, 2.93, 9.77, 0.39].';
a1_8000 = 0.9614;  
b1_8000 = 60.2912;  
c1_8000 = 101.9641;  
% rep percent thresh 39:
y1_9000 = [88.38, 53.61, 7.13, 18.16, 2.34].';
a1_9000 = 0.9601; 
b1_9000 = 44.3049;
c1_9000 = 101.9366;
% rep percent thresh 42:
y1_10000 = [90.14, 64.36, 15.04, 34.67, 5.27].';
a1_10000 = 0.9587;  
b1_10000 = 32.2278;
c1_10000 = 101.2248;
% rep percent thresh 33:
y1_7000 = [84.57, 23.05, 0.49, 1.76, 0.20].'; 
a1_7000 = 0.9636; 
b1_7000 = 90.1602;
c1_7000 = 101.9023;

%% (2) stylegan2 FLOWER:
x2 = [22.02, 27.41, 30.34]'; % ID
x2_norm = (x2-x_mean_common) / x_std_common;
xx2 = linspace(20,35,50); % for x = ID
% rep percent thresh 36:
y2_8000 = [32.13, 2.83, 1.95]';  
a2_8000 = 0.9719;
b2_8000 = 100.0000;
c2_8000 = 102.0000;
% rep percent thresh 39:
y2_9000 = [45.61, 8.11, 4.79]';
a2_9000 = 0.9670;
b2_9000 = 60.0000;
c2_9000 = 102.0000;
% rep percent thresh 42:
y2_10000 = [57.91, 14.94, 10.64]';
a2_10000 = 0.9635;
b2_10000 = 39.3434;
c2_10000 = 101.2707;
% rep percent thresh 33:
y2_7000 = [18.46, 0.59, 0.49]';
a2_7000 = 0.9796;
b2_7000 = 200.0000;
c2_7000 = 102.0000;

%%
x_all = {x1,x2};
x_norm_all = {x1_norm,x2_norm};
xx_all = {xx1,xx2};
% rep percent thresh 36:
y_8000_all = {y1_8000,y2_8000};
a_8000_all = {a1_8000,a2_8000};
b_8000_all = {b1_8000,b2_8000};
c_8000_all = {c1_8000,c2_8000};
% rep percent thresh 39:
y_9000_all = {y1_9000,y2_9000};
a_9000_all = {a1_9000,a2_9000};
b_9000_all = {b1_9000,b2_9000};
c_9000_all = {c1_9000,c2_9000};
% rep percent thresh 42:
y_10000_all = {y1_10000,y2_10000};
a_10000_all = {a1_10000,a2_10000};
b_10000_all = {b1_10000,b2_10000};
c_10000_all = {c1_10000,c2_10000};
% rep percent thresh 33:
y_7000_all = {y1_7000,y2_7000};
a_7000_all = {a1_7000,a2_7000};
b_7000_all = {b1_7000,b2_7000};
c_7000_all = {c1_7000,c2_7000};

title_list = {'biggan FLOWER', 'stylegan2 FLOWER'};

for i = 1:n_case % for each GAN curve
    this_title = title_list{i};
    x_real = x_all{i};
    %xx = xx_all{i};
    
    h = zeros(4,1);
    % rep percent thresh 33:
    y = y_7000_all{i};
    a = a_7000_all{i};
    b = b_7000_all{i};
    c = c_7000_all{i};
    syms x
    f = a^(b*(x-x_mean_common)/x_std_common-c);
    figure;
    h(1) = fplot(f, '-','color',rgb('RoyalBlue'),'LineWidth', 3);
    hold on
    plot(x_real,y,'.','color',rgb('RoyalBlue'),'MarkerSize',60);
    hold on
    % rep percent thresh 36:
    y = y_8000_all{i};
    a = a_8000_all{i};
    b = b_8000_all{i};
    c = c_8000_all{i};
    syms x
    f = a^(b*(x-x_mean_common)/x_std_common-c);
    h(2) = fplot(f, '-','color',rgb('Turquoise'),'LineWidth', 3);
    hold on
    plot(x_real,y,'.','color',rgb('Turquoise'),'MarkerSize',60);
    hold on
    % rep percent thresh 39:
    y = y_9000_all{i};
    a = a_9000_all{i};
    b = b_9000_all{i};
    c = c_9000_all{i};
    syms x
    f = a^(b*(x-x_mean_common)/x_std_common-c);
    h(3) = fplot(f, '-','color',rgb('YellowGreen'),'LineWidth', 3);
    hold on
    plot(x_real,y,'.','color',rgb('YellowGreen'),'MarkerSize',60);
    hold on
    % rep percent thresh 42:
    y = y_10000_all{i};
    a = a_10000_all{i};
    b = b_10000_all{i};
    c = c_10000_all{i};
    syms x
    f = a^(b*(x-x_mean_common)/x_std_common-c);
    h(4) = fplot(f, '-','color',rgb('Gold'),'LineWidth', 3);
    hold on
    plot(x_real,y,'.','color',rgb('Gold'),'MarkerSize',60);
    
    grid on;
    title(this_title,'FontSize', 20);
    xlabel('ID', 'FontSize', 18);
    ylabel('replication percent', 'FontSize', 18');
    ylim([0 100])
    yticks(0 : 20 : 100);
    xx_tmp = xx_all{i};
    xlim([min(xx_tmp) max(xx_tmp)])
    xticks(min(xx_tmp) : 5 : max(xx_tmp));
    % Get handle to current axes.
    ax = gca;
    % Set x and y font sizes.
    ax.XAxis.FontSize = 18;
    ax.YAxis.FontSize = 18;
    
    lgd = legend(h, '\alpha=33','\alpha=36','\alpha=39','\alpha=42');
    lgd.FontSize = 15;
    hold off;


end
