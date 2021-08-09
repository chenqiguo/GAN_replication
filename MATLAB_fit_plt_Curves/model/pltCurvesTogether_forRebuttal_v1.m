% plot all the required curves together in one figure for rebuttal:
% version 1: stylegan_FLOWER

%% parameters:
%n_iter_fit = 20; % num of iterations for each a,b,c estimate
n_case = 8; %7; % num of GAN curves
%n_test = 1; % num of testing pairs
% an array of all the dataset ID values:
x_tmp = [22.02, 24.70, 27.41, 28.99, 30.34,...
         11.90, 15.97, 17.30, 21.34, 23.29,...
         14.87, 20.80, 27.06, 29.57, 33.60];
x_mean_common = mean(x_tmp);
x_std_common = std(x_tmp);


%% (1) stylegan2 FLOWER original thresh=8000 (in paper):
x1 = [22.02, 27.41, 30.34]'; % ID
x1_norm = (x1-x_mean_common) / x_std_common;
xx1 = linspace(20,35,50); % for x = ID
% rep percent thresh 8000:
y1 = [31.93, 1.76, 1.07]';  
a1_8000 = 0.9723;
b1_8000 = 116.3858;
c1_8000 = 100.0763;

%% (2) stylegan2 FLOWER new metric (combination of inceptionv3 & pixel-wise) thresh=36 (new experiment):
x2 = [22.02, 27.41, 30.34]'; % ID
x2_norm = (x2-x_mean_common) / x_std_common;
xx2 = linspace(20,35,50); % for x = ID
y2 = [32.13, 2.83, 1.95]';  % rep percent thresh 36
a2_36 = 0.9719;
b2_36 = 100.0000;
c2_36 = 102.0000;

%% (3) stylegan2 CIFAR10 thresh=2000 (new experiment):
x3 = [17.89, 23.32, 25.73, 26.49]'; % ID
x3_norm = (x3-x_mean_common) / x_std_common;
y3 = [24.02, 17.97, 16.70, 17.58].'; % rep percent thresh 2000
xx3 = linspace(15,30,50); % for x = ID
a3_2000 = 0.9713;
b3_2000 = 9.1261;
c3_2000 = 101.0888;

%% (4) stylegan2 FLOWER new k1&k2 values thresh=8000 (new experiment):
x4 = [20.23, 25.61, 28.43]'; % ID
x4_norm = (x4-x_mean_common) / x_std_common;
y4 = [31.93, 1.76, 1.07]';  % rep percent thresh 8000
xx4 = linspace(20,30,50); % for x = ID
a4_8000 = 0.9813;
b4_8000 = 173.6261;
c4_8000 = 100.0000;

%% (5) stylegan2 CelebA original thresh=8000 (new experiment):
x5 = [17.30, 21.34, 23.29]'; % ID
x5_norm = (x5-x_mean_common) / x_std_common;
xx5 = linspace(10,25,50); % for x = ID
% rep percent thresh 8000:
y5 = [76.86, 21.19, 15.14]'; %97.36, 
a5_8000 = 0.9751;
b5_8000 = 70.0000;
c5_8000 = 101.0000;

%% (6) stylegan2 CelebA the added new sub-level thresh=8000 (new experiment):
x6 = 15.97; % ID
x6_norm = (x6-x_mean_common) / x_std_common;
xx6 = linspace(10,25,50); % for x = ID
% rep percent thresh 8000:
y6 = 91.02;
% dummy values:
a6_ = -1;
b6_ = -1;
c6_ = -1;

%% (7) stylegan2 FLOWER new resolution 256x256 thresh=23000 (new experiment):
x7 = [22.02, 27.41, 28.99, 30.34]'; % ID
x7_norm = (x7-x_mean_common) / x_std_common;
xx7 = linspace(20,35,50); % for x = ID
% rep percent thresh 23000:
y7 = [24.02, 14.94, 15.63, 15.33]';
a7_23000 = 0.9699;
b7_23000 = 12.8175;
c7_23000 = 100.9197;

%% (8) stylegan2 FLOWER 128x128 early iteration thresh=10000:
x8 = [22.02, 27.41, 30.34]'; % ID
x8_norm = (x8-x_mean_common) / x_std_common;
y8 = [20.61, 11.23, 9.67]';  % rep percent thresh 10000
xx8 = linspace(20,35,50); % for x = ID
a8_10000 = 0.9718;
b8_10000 = 21.9088;
c8_10000 = 101.2936;

%%
x_all = {x1,x2,x3,x4,x5,x6,x7,x8};
x_norm_all = {x1_norm,x2_norm,x3_norm,x4_norm,x5_norm,x6_norm,x7_norm,x8_norm};
xx_all = {xx1,xx2,xx3,xx4,xx5,xx6,xx7,xx8};

y_all = {y1,y2,y3,y4,y5,y6,y7,y8};
a_all = {a1_8000,a2_36,a3_2000,a4_8000,a5_8000,a6_,a7_23000,a8_10000};
b_all = {b1_8000,b2_36,b3_2000,b4_8000,b5_8000,b6_,b7_23000,b8_10000};
c_all = {c1_8000,c2_36,c3_2000,c4_8000,c5_8000,c6_,c7_23000,c8_10000};

%title_list = {'stylegan2 FLOWER original', 'stylegan2 FLOWER new metric',...
%              'stylegan2 CIFAR10', 'stylegan2 FLOWER new k1k2',...
%              'stylegan2 CelebA new sub-levels'};%,'stylegan2 FLOWER new resolution256'
color_list = {'DeepPink', 'Gold', 'RoyalBlue', 'Turquoise', 'YellowGreen',...
              'YellowGreen', 'DarkViolet', 'Silver'};

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
    
    if i~=6 % for NOT 'star'
        syms x
        f = a^(b*(x-x_mean_common)/x_std_common-c);
        h(i) = fplot(f, '-','color', rgb(this_color),'LineWidth', 3);
        hold on
    end
    
    if i==6 % only for the 'star'
        % plot star at the added point:
        h(i) = plot(x_real,y, 'p', 'color', rgb(this_color), 'MarkerFaceColor', rgb(this_color), 'MarkerSize',30);
        hold on
        % plot star at (13,70) to be removed to legend later:
        plot((13-x_mean_common)/x_std_common-c,70, 'p', 'color', rgb(this_color), 'MarkerFaceColor', rgb(this_color), 'MarkerSize',30);
    else
        %plot(x_real,y,'.','color',rgb(this_color),'MarkerSize',60);
        scatter_plt = scatter(x_real,y, 280, 'MarkerFaceColor',rgb(this_color),'MarkerEdgeColor',rgb(this_color));
        scatter_plt.MarkerFaceAlpha = .65;
        scatter_plt.MarkerEdgeAlpha = .65;
    end
    hold on

end

grid on;
title('StyleGAN2 fitting curves (for rebuttal)','FontSize', 20);
xlabel('ID', 'FontSize', 18);
ylabel('replication percent', 'FontSize', 18');
ylim([0 100])
yticks(0 : 20 : 100);
xx_tmp = linspace(10,35,50);
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
lgd = legend(h, '1. FLOWER original', '2. FLOWER combined metric',...
                '3. CIFAR10', '4. FLOWER k_1=20 k_2=30',...
                '5. CelebA original', '6.  new sublevel = 600',...
                '7. FLOWER resolution = 256', '7. FLOWER resolution = 128 early itr');

lgd.FontSize = 15;
hold off;

