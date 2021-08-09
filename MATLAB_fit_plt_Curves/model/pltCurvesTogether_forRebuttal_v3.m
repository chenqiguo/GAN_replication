% make a figure same as Figure A1(a) in our rebuttal but only with C1 and C2 curves

%% parameters:
%n_iter_fit = 20; % num of iterations for each a,b,c estimate
n_case1 = 2; %5; % num of GAN curves for the five upper
%n_case2 = 3; % num of GAN curves for the three lower
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

%%
x_all = {x1,x2};
x_norm_all = {x1_norm,x2_norm};
xx_all = {xx1,xx2};

y_all = {y1,y2};
a_all = {a1_8000,a2_36};
b_all = {b1_8000,b2_36};
c_all = {c1_8000,c2_36};

color_list = {'DeepPink', 'Gold'};

%% for the five upper:
figure;
h = zeros(n_case1,1);
for i = 1:n_case1 % for each GAN curve
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
title('StyleGAN2 FLOWER different metrics','FontSize', 20);
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
lgd = legend(h, '1. FLOWER original', '2. FLOWER combined metric');

lgd.FontSize = 15;
hold off;


