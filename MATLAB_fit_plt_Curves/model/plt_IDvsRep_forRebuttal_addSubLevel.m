% ONLY for additional subset levels (for rebuttal):

% plot scatters & fitted curves of ID vs replication,
% for threshold = 7000, 8000, 9000, 10000.
% This is for rebuttal.

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

%% (4) stylegan2 CelebA:
x4 = [15.97, 17.30, 21.34, 23.29]'; %11.90,  ID
x4_norm = (x4-x_mean_common) / x_std_common;
xx4 = linspace(10,25,50); % for x = ID
% rep percent thresh 8000:
y4_8000 = [91.02, 76.86, 21.19, 15.14]'; %97.36, 
a4_8000 = 0.9751;
b4_8000 = 70.0000;
c4_8000 = 101.0000;
% rep percent thresh 9000:
y4_9000 = [96.29, 91.41, 44.43, 33.01]'; %98.54, 
a4_9000 = 0.9666;
b4_9000 = 30.0000;
c4_9000 = 102.0000;
% rep percent thresh 10000:
y4_10000 = [98.83, 98.05, 67.77, 54.39]'; %99.51, 
a4_10000 = 0.9614;
b4_10000 = 15.0000;
c4_10000 = 101.0000;
% rep percent thresh 7000:
y4_7000 = [79.00, 54.00, 4.59, 3.91]'; %95.51, 
a4_7000 = 0.9934;
b4_7000 = 490.0000;
c4_7000 = 101.0000;

%% (5) biggan LSUN:
x5 = [14.87, 20.80, 27.06, 29.57, 33.60]'; % ID
x5_norm = (x5-x_mean_common) / x_std_common;
xx5 = linspace(10,35,50); % for x = ID
% rep percent thresh 8000:
y5_8000 = [38.57, 27.15, 0, 0.20, 0.20]';
a5_8000 = 0.9781;
b5_8000 = 50.0000;
c5_8000 = 102.0000;
% rep percent thresh 9000:
y5_9000 = [66.41, 56.54, 0.39, 1.37, 1.76]';
a5_9000 = 0.9692;
b5_9000 = 26.3826;
c5_9000 = 101.7170;
% rep percent thresh 10000:
y5_10000 = [88.57, 76.17, 4.10, 8.50, 6.15]'; 
a5_10000 = 0.9650;
b5_10000 = 21.1384;
c5_10000 = 100.3657;
% rep percent thresh 7000:
y5_7000 = [18.95, 8.79, 0, 0, 6.15]'; 
a5_7000 = 0.9838;
b5_7000 = 60.3867;
c5_7000 = 100.0000;

%%
x_all = {x4,x5};
x_norm_all = {x4_norm,x5_norm};
xx_all = {xx4,xx5};
% rep percent thresh 8000:
y_8000_all = {y4_8000,y5_8000};
a_8000_all = {a4_8000,a5_8000};
b_8000_all = {b4_8000,b5_8000};
c_8000_all = {c4_8000,c5_8000};
% rep percent thresh 9000:
y_9000_all = {y4_9000,y5_9000};
a_9000_all = {a4_9000,a5_9000};
b_9000_all = {b4_9000,b5_9000};
c_9000_all = {c4_9000,c5_9000};
% rep percent thresh 10000:
y_10000_all = {y4_10000,y5_10000};
a_10000_all = {a4_10000,a5_10000};
b_10000_all = {b4_10000,b5_10000};
c_10000_all = {c4_10000,c5_10000};
% rep percent thresh 7000:
y_7000_all = {y4_7000,y5_7000};
a_7000_all = {a4_7000,a5_7000};
b_7000_all = {b4_7000,b5_7000};
c_7000_all = {c4_7000,c5_7000};

title_list = {'stylegan2 CelebA', 'biggan LSUN'};

for i = 1:n_case % for each GAN curve
    this_title = title_list{i};
    x_real = x_all{i};
    %xx = xx_all{i};
    
    h = zeros(4,1);
    % rep percent thresh 7000:
    y = y_7000_all{i};
    a = a_7000_all{i};
    b = b_7000_all{i};
    c = c_7000_all{i};
    syms x
    f = a^(b*(x-x_mean_common)/x_std_common-c);
    figure;
    h(1) = fplot(f, '-','color',rgb('RoyalBlue'),'LineWidth', 3);
    hold on
    if strcmp(this_title, 'stylegan2 CelebA')
        plot(x_real(2:end),y(2:end),'.','color',rgb('RoyalBlue'),'MarkerSize',60);
        hold on
        plot(x_real(1),y(1), 'p', 'color', rgb('RoyalBlue'), 'MarkerFaceColor', rgb('RoyalBlue'), 'MarkerSize',30);
    else
        plot(x_real(1:end-1),y(1:end-1),'.','color',rgb('RoyalBlue'),'MarkerSize',60);
        hold on
        plot(x_real(end),y(end),'p','color',rgb('RoyalBlue'),'MarkerFaceColor',rgb('RoyalBlue'), 'MarkerSize',30);
    end
    hold on
    % rep percent thresh 8000:
    y = y_8000_all{i};
    a = a_8000_all{i};
    b = b_8000_all{i};
    c = c_8000_all{i};
    syms x
    f = a^(b*(x-x_mean_common)/x_std_common-c);
    h(2) = fplot(f, '-','color',rgb('Turquoise'),'LineWidth', 3);
    hold on
    if strcmp(this_title, 'stylegan2 CelebA')
        plot(x_real(2:end),y(2:end),'.','color',rgb('Turquoise'),'MarkerSize',60);
        hold on
        plot(x_real(1),y(1),'p','color',rgb('Turquoise'),'MarkerFaceColor',rgb('Turquoise'), 'MarkerSize',30);
    else
        plot(x_real(1:end-1),y(1:end-1),'.','color',rgb('Turquoise'),'MarkerSize',60);
        hold on
        plot(x_real(end),y(end),'p','color',rgb('Turquoise'),'MarkerFaceColor',rgb('Turquoise'), 'MarkerSize',30);
    end
    hold on
    % rep percent thresh 9000:
    y = y_9000_all{i};
    a = a_9000_all{i};
    b = b_9000_all{i};
    c = c_9000_all{i};
    syms x
    f = a^(b*(x-x_mean_common)/x_std_common-c);
    h(3) = fplot(f, '-','color',rgb('YellowGreen'),'LineWidth', 3);
    hold on
    if strcmp(this_title, 'stylegan2 CelebA')
        plot(x_real(2:end),y(2:end),'.','color',rgb('YellowGreen'),'MarkerSize',60);
        hold on
        plot(x_real(1),y(1),'p','color',rgb('YellowGreen'),'MarkerFaceColor',rgb('YellowGreen'), 'MarkerSize',30);
    else
        plot(x_real(1:end-1),y(1:end-1),'.','color',rgb('YellowGreen'),'MarkerSize',60);
        hold on
        plot(x_real(end),y(end),'p','color',rgb('YellowGreen'),'MarkerFaceColor',rgb('YellowGreen'), 'MarkerSize',30);
    end
    hold on
    % rep percent thresh 10000:
    y = y_10000_all{i};
    a = a_10000_all{i};
    b = b_10000_all{i};
    c = c_10000_all{i};
    syms x
    f = a^(b*(x-x_mean_common)/x_std_common-c);
    h(4) = fplot(f, '-','color',rgb('Gold'),'LineWidth', 3);
    hold on
    if strcmp(this_title, 'stylegan2 CelebA')
        plot(x_real(2:end),y(2:end),'.','color',rgb('Gold'),'MarkerSize',60);
        hold on
        plot(x_real(1),y(1),'p','color',rgb('Gold'),'MarkerFaceColor',rgb('Gold'), 'MarkerSize',30);
    else
        plot(x_real(1:end-1),y(1:end-1),'.','color',rgb('Gold'),'MarkerSize',60);
        hold on
        plot(x_real(end),y(end),'p','color',rgb('Gold'),'MarkerFaceColor',rgb('Gold'), 'MarkerSize',30);
    end
    
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
    
    lgd = legend(h, '\alpha=7000','\alpha=8000','\alpha=9000','\alpha=10000');
    lgd.FontSize = 15;
    hold off;
    
end
