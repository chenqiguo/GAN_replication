% plot curves (thresh8000) for the FLOWER, CelebA and LSUN dataset,
% and use scatter for the MNIST (thresh8000),
% separately for StyleGAN2 and BigGAN.

%% parameters:
%n_iter_fit = 20; % num of iterations for each a,b,c estimate
n_case1 = 4; % num of curves for the BigGAN
n_case2 = 4; % num of curves for the StyleGAN2
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
% rep percent thresh 8000:
y1_8000 = [76.07, 34.77, 1.86, 4.49, 0.29].';
a1_8000 = 0.9621;  
b1_8000 = 62.9262;  
c1_8000 = 100.0000;  

%% (2) stylegan2 FLOWER:
x2 = [22.02, 27.41, 30.34]'; % ID
x2_norm = (x2-x_mean_common) / x_std_common;
xx2 = linspace(20,35,50); % for x = ID
% rep percent thresh 8000:
y2_8000 = [31.93, 1.76, 1.07]';  
a2_8000 = 0.9723;
b2_8000 = 116.3858;
c2_8000 = 100.0763;

%% (3) biggan CelebA:
x3 = [11.90, 15.97, 17.30, 21.34, 23.29]'; % ID
x3_norm = (x3-x_mean_common) / x_std_common;
xx3 = linspace(10,25,50); % for x = ID
% rep percent thresh 8000:
y3_8000 = [84.18, 19.82, 4.30, 11.43, 6.05]'; 
a3_8000 = 0.9869; 
b3_8000 = 130.0000; 
c3_8000 = 100.0000;

%% (4) stylegan2 CelebA:
x4 = [17.30, 21.34, 23.29]'; % ID
x4_norm = (x4-x_mean_common) / x_std_common;
xx4 = linspace(15,25,50); % for x = ID
% rep percent thresh 8000:
y4_8000 = [76.86, 21.19, 15.14]'; 
a4_8000 = 0.9747;
b4_8000 = 73.6460;
c4_8000 = 100.0000;

%% (5) biggan LSUN:
x5 = [14.87, 20.80, 27.06, 29.57]'; % ID
x5_norm = (x5-x_mean_common) / x_std_common;
xx5 = linspace(10,35,50); % for x = ID
% rep percent thresh 8000:
y5_8000 = [38.57, 27.15, 0, 0.20]';
a5_8000 = 0.9756;
b5_8000 = 36.4029;
c5_8000 = 101.6608;

%% (6) stylegan2 LSUN:
x6 = [14.87, 20.80, 27.06, 29.57, 33.60]'; % ID
x6_norm = (x6-x_mean_common) / x_std_common;
xx6 = linspace(10,35,50); % for x = ID
% rep percent thresh 8000:
y6_8000 = [92.38, 27.93, 1.17, 3.03, 4.30]'; 
a6_8000 = 0.9735;  
b6_8000 = 51.5465;  
c6_8000 = 100.3704;

%% (7) biggan MNIST:
x7 = [11.86, 12.30, 12.59]'; % ID
x7_norm = (x7-x_mean_common) / x_std_common; % ???
xx7 = linspace(11.5,13.3,50); % for x = ID
% rep percent thresh 8000:
y7_8000 = [49.22, 32.91, 47.07].';
a7_8000 = 0; % dummy val
b7_8000 = 0; % dummy val 
c7_8000 = 0; % dummy val

%% (8) stylegan2 MNIST:
x8 = [11.86, 12.30, 12.59]'; % ID
x8_norm = (x8-x_mean_common) / x_std_common; % ???
xx8 = linspace(11.5,13.3,50); % for x = ID
% rep percent thresh 8000:
y8_8000 = [61.33, 66.80, 70.31]';
a8_8000 = 0; % dummy val
b8_8000 = 0; % dummy val 
c8_8000 = 0; % dummy val

%%
color_list = {'DeepPink', 'Gold', 'Turquoise', 'YellowGreen'};

%% for biggan:
x_all_1 = {x1,x3,x5,x7};
x_norm_all_1 = {x1_norm,x3_norm,x5_norm,x7_norm};
xx_all_1 = {xx1,xx3,xx5,xx7};
% rep percent thresh 8000:
y_8000_all_1 = {y1_8000,y3_8000,y5_8000,y7_8000};
a_8000_all_1 = {a1_8000,a3_8000,a5_8000,a7_8000};
b_8000_all_1 = {b1_8000,b3_8000,b5_8000,b7_8000};
c_8000_all_1 = {c1_8000,c3_8000,c5_8000,c7_8000};

figure;
h = zeros(n_case1,1);
for i = 1:n_case1 % for each biggan curve
    this_color = color_list{i};
    x_real = x_all_1{i};
    %xx = xx_all_1{i};
    
    y = y_8000_all_1{i};
    a = a_8000_all_1{i};
    b = b_8000_all_1{i};
    c = c_8000_all_1{i};
    
    if i~=4 % for NOT MNIST
        syms x
        f = a^(b*(x-x_mean_common)/x_std_common-c);
        h(i) = fplot(f, '-','color', rgb(this_color),'LineWidth', 3);
        hold on
    else
        h(i) = plot(x_real,y,'-','color',rgb(this_color),'LineWidth', 3);
    end
    
    %plot(x_real,y,'.','color',rgb(this_color),'MarkerSize',60);
    scatter_plt = scatter(x_real,y, 280, 'MarkerFaceColor',rgb(this_color),'MarkerEdgeColor',rgb(this_color));
    scatter_plt.MarkerFaceAlpha = .65;
    scatter_plt.MarkerEdgeAlpha = .65;
    
    hold on

end

grid on;
title('BigGAN fitting curves with \alpha=8000','FontSize', 20);
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
lgd = legend(h, '1. FLOWER', '2. CelebA',...
                '3. LSUN', '4. MNIST');

lgd.FontSize = 15;
hold off;

%% for stylegan:
x_all_2 = {x2,x4,x6,x8};
x_norm_all_2 = {x2_norm,x4_norm,x6_norm,x8_norm};
xx_all_2 = {xx2,xx4,xx6,xx8};
% rep percent thresh 8000:
y_8000_all_2 = {y2_8000,y4_8000,y6_8000,y8_8000};
a_8000_all_2 = {a2_8000,a4_8000,a6_8000,a8_8000};
b_8000_all_2 = {b2_8000,b4_8000,b6_8000,b8_8000};
c_8000_all_2 = {c2_8000,c4_8000,c6_8000,c8_8000};

figure;
h = zeros(n_case2,1);
for i = 1:n_case2 % for each stylegan curve
    this_color = color_list{i};
    x_real = x_all_2{i};
    %xx = xx_all_2{i};
    
    y = y_8000_all_2{i};
    a = a_8000_all_2{i};
    b = b_8000_all_2{i};
    c = c_8000_all_2{i};
    
    if i~=4 % for NOT MNIST
        syms x
        f = a^(b*(x-x_mean_common)/x_std_common-c);
        h(i) = fplot(f, '-','color', rgb(this_color),'LineWidth', 3);
        hold on
    else
        h(i) = plot(x_real,y,'-','color',rgb(this_color),'LineWidth', 3);
    end
    
    %plot(x_real,y,'.','color',rgb(this_color),'MarkerSize',60);
    scatter_plt = scatter(x_real,y, 280, 'MarkerFaceColor',rgb(this_color),'MarkerEdgeColor',rgb(this_color));
    scatter_plt.MarkerFaceAlpha = .65;
    scatter_plt.MarkerEdgeAlpha = .65;
    
    hold on

end

grid on;
title('StyleGAN2 fitting curves with \alpha=8000','FontSize', 20);
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
lgd = legend(h, '1. FLOWER', '2. CelebA',...
                '3. LSUN', '4. MNIST');

lgd.FontSize = 15;
hold off;






