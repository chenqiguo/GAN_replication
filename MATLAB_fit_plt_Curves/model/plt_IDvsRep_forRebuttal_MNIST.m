% ONLY for MNIST dataset (for rebuttal):

% plot scatters of ID vs replication,
% for threshold = 7000, 8000, 9000, 10000.
% This is for rebuttal.

% NOTE: not plot fitted curves yet --> since MNIST has trend different from
% other datasets!

% referenced from plt_IDvsRep_forSuppMat.m

%% parameters:
% an array of all the MNIST dataset ID values:
x_tmp = [11.86, 12.30, 12.59];
x_mean_common = mean(x_tmp);
x_std_common = std(x_tmp);

%% (1) biggan MNIST:
x1 = [11.86, 12.30, 12.59]'; % ID
x1_norm = (x1-x_mean_common) / x_std_common;
xx1 = linspace(11.5,13.3,50); % for x = ID
% rep percent thresh 8000:
y1_8000 = [49.22, 32.91, 47.07].';
% rep percent thresh 9000:
y1_9000 = [64.75, 47.75, 62.79].';
% rep percent thresh 10000:
y1_10000 = [78.32, 63.96, 76.95].';
% rep percent thresh 7000:
y1_7000 = [34.28, 20.02, 31.84].'; 

%% (2) stylegan2 MNIST:
x2 = [11.86, 12.30, 12.59]'; % ID
x2_norm = (x2-x_mean_common) / x_std_common;
xx2 = linspace(11.5,13.3,50); % for x = ID
% rep percent thresh 8000:
y2_8000 = [61.33, 66.80, 70.31]';
% rep percent thresh 9000:
y2_9000 = [79.00, 83.98, 85.74]';
% rep percent thresh 10000:
y2_10000 = [91.50, 92.58, 94.34]';
% rep percent thresh 7000:
y2_7000 = [42.09, 45.41, 51.07]';

%%
x_all = {x1,x2};
x_norm_all = {x1_norm,x2_norm};
xx_all = {xx1,xx2};
% rep percent thresh 8000:
y_8000_all = {y1_8000,y2_8000};
% rep percent thresh 9000:
y_9000_all = {y1_9000,y2_9000};
% rep percent thresh 10000:
y_10000_all = {y1_10000,y2_10000};
% rep percent thresh 7000:
y_7000_all = {y1_7000,y2_7000};

title_list = {'biggan MNIST', 'stylegan2 MNIST'};

for i = 1:2 % for each MNIST GAN curve
    this_title = title_list{i};
    x_real = x_all{i};
    %xx = xx_all{i};
    
    h = zeros(4,1);
    % rep percent thresh 7000:
    y = y_7000_all{i};
    figure;
    h(1) = plot(x_real,y,'.-','color',rgb('RoyalBlue'),'MarkerSize',60);
    hold on
    % rep percent thresh 8000:
    y = y_8000_all{i};
    h(2) = plot(x_real,y,'.-','color',rgb('Turquoise'),'MarkerSize',60);
    hold on
    % rep percent thresh 9000:
    y = y_9000_all{i};
    h(3) = plot(x_real,y,'.-','color',rgb('YellowGreen'),'MarkerSize',60);
    hold on
    % rep percent thresh 10000:
    y = y_10000_all{i};
    h(4) = plot(x_real,y,'.-','color',rgb('Gold'),'MarkerSize',60);
    
    grid on;
    title(this_title,'FontSize', 20);
    xlabel('ID', 'FontSize', 18);
    ylabel('replication percent', 'FontSize', 18');
    ylim([0 100])
    yticks(0 : 20 : 100);
    xx_tmp = xx_all{i};
    xlim([min(xx_tmp) max(xx_tmp)])
    xticks(min(xx_tmp) : 0.3 : max(xx_tmp));
    % Get handle to current axes.
    ax = gca;
    % Set x and y font sizes.
    ax.XAxis.FontSize = 18;
    ax.YAxis.FontSize = 18;
    
    lgd = legend(h, '\alpha=7000','\alpha=8000','\alpha=9000','\alpha=10000');
    lgd.FontSize = 15;
    hold off;
    
    
end



