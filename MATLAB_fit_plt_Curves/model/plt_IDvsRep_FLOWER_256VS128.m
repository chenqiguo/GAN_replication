% plot together: compare the FLOWER_256_xxx (thresh18000) curve to
% the FLOWER_128_xxx (thresh8000) version:

%% parameters:
n_case = 2; % num of GAN curves
% an array of all the dataset ID values:
x_tmp = [22.02, 24.70, 27.41, 28.99, 30.34,...
         11.90, 15.97, 17.30, 21.34, 23.29,...
         14.87, 20.80, 27.06, 29.57, 33.60];
x_mean_common = mean(x_tmp);
x_std_common = std(x_tmp);

%% (1) stylegan2 FLOWER_256_xxx:
x1 = [22.02, 27.41, 28.99, 30.34]'; % ID
x1_norm = (x1-x_mean_common) / x_std_common;
xx1 = linspace(20,35,50); % for x = ID
% rep percent thresh 18000:
y1_18000 = [48.13, 15.42, 11.04, 9.38]';  % final ver: rep percent thresh 18000
a1_18000 = 0.9647;
b1_18000 = 36.5150;
c1_18000 = 100.5885;

%% (2) stylegan2 FLOWER_128_xxx:
x2 = [22.02, 27.41, 30.34]'; % ID
x2_norm = (x2-x_mean_common) / x_std_common;
xx2 = linspace(20,35,50); % for x = ID
% rep percent thresh 8000:
y2_8000 = [31.93, 1.76, 1.07]';  
a2_8000 = 0.9723;
b2_8000 = 116.3858;
c2_8000 = 100.0763;

%%
x_all = {x2,x1};
x_norm_all = {x2_norm,x1_norm};
xx_all = {xx2,xx1};
% rep percent:
y_all = {y2_8000,y1_18000};
a_all = {a2_8000,a1_18000};
b_all = {b2_8000,b1_18000};
c_all = {c2_8000,c1_18000};

color_list = {'Silver','DarkViolet'};

figure;
h = zeros(n_case,1);
for i = 1:n_case % for each GAN curve
    %this_title = title_list{i};
    this_color = color_list{i};
    x_real = x_all{i};
    %xx = xx_all{i};
    
    y = y_all{i};
    a = a_all{i};
    b = b_all{i};
    c = c_all{i};
    
    syms x
    f = a^(b*(x-x_mean_common)/x_std_common-c);
    h(i) = fplot(f, '-','color', rgb(this_color),'LineWidth', 3);
    hold on
    
    %plot(x_real,y,'.','color',rgb(this_color),'MarkerSize',60);
    scatter_plt = scatter(x_real,y, 280, 'MarkerFaceColor',rgb(this_color),'MarkerEdgeColor',rgb(this_color));
    scatter_plt.MarkerFaceAlpha = .65;
    scatter_plt.MarkerEdgeAlpha = .65;
    
    hold on

end

grid on;
title('StyleGAN2 FLOWER resolution 128 vs 256','FontSize', 20);
xlabel('ID', 'FontSize', 18);
ylabel('replication percent', 'FontSize', 18');
ylim([0 100])
yticks(0 : 20 : 100);
xx_tmp = linspace(15,35,50);
xlim([min(xx_tmp) max(xx_tmp)])
xticks(min(xx_tmp) : 5 : max(xx_tmp));
% Get handle to current axes.
ax = gca;
% Set x and y font sizes.
ax.XAxis.FontSize = 18;
ax.YAxis.FontSize = 18;

%lgd = legend(h, 'FLOWER original \alpha=8000', 'FLOWER combined metric \alpha=36',...
%                'CIFAR10 \alpha=2000', 'FLOWER k_1=20 k_2=30 \alpha=8000',...
%                'CelebA adding sub-level \alpha=8000');%,'FLOWER resolution=256 \alpha=???'
lgd = legend(h, '6. FLOWER resolution = 128',...
                '7. FLOWER resolution = 256');

lgd.FontSize = 15;
hold off;

