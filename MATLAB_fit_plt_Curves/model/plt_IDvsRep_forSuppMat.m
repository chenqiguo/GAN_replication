% plot scatters & fitted curves of ID vs replication,
% for threshold = 7000, 8000, 9000, 10000.
% This is for supplemental material.


%% parameters:
n_iter_fit = 20; % num of iterations for each a,b,c estimate
n_case = 6; % num of GAN curves
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
% rep percent thresh 8000:
y1_8000 = [76.07, 34.77, 1.86, 4.49, 0.29].';
a1_8000 = 0.9621;  
b1_8000 = 62.9262;  
c1_8000 = 100.0000;  
% rep percent thresh 9000:
y1_9000 = [85.55, 53.71, 6.74, 17.97, 1.56].';
a1_9000 = 0.9595; 
b1_9000 = 43.1907;
c1_9000 = 100.0000;
% rep percent thresh 10000:
y1_10000 = [89.75, 65.82, 16.89, 36.43, 5.37].';
a1_10000 = 0.9582;  
b1_10000 = 30.5225;
c1_10000 = 100.1540;
% rep percent thresh 7000:
y1_7000 = [66.11, 18.95, 0.10, 1.17, 0].'; 
a1_7000 = 0.9652; 
b1_7000 = 92.3603;
c1_7000 = 100.0000;

%% (2) stylegan2 FLOWER:
x2 = [22.02, 27.41, 30.34]'; % ID
x2_norm = (x2-x_mean_common) / x_std_common;
xx2 = linspace(20,35,50); % for x = ID
% rep percent thresh 8000:
y2_8000 = [31.93, 1.76, 1.07]';  
a2_8000 = 0.9723;
b2_8000 = 116.3858;
c2_8000 = 100.0763;
% rep percent thresh 9000:
y2_9000 = [47.66, 7.42, 4.00]';
a2_9000 = 0.9663;
b2_9000 = 61.3039;
c2_9000 = 100.6188;
% rep percent thresh 10000:
y2_10000 = [63.09, 16.31, 10.74]';
a2_10000 = 0.9625;
b2_10000 = 38.9430;
c2_10000 = 100.5178;
% rep percent thresh 7000:
y2_7000 = [15.53, 0.29, 0.20]';
a2_7000 = 0.9818;
b2_7000 = 248.3521;
c2_7000 = 100.0002;

%% (3) biggan CelebA:
x3 = [11.90, 15.97, 17.30, 21.34, 23.29]'; % ID
x3_norm = (x3-x_mean_common) / x_std_common;
xx3 = linspace(10,25,50); % for x = ID
% rep percent thresh 8000:
y3_8000 = [84.18, 19.82, 4.30, 11.43, 6.05]'; 
a3_8000 = 0.9869; 
b3_8000 = 130.0000; 
c3_8000 = 100.0000;
% rep percent thresh 9000:
y3_9000 = [92.09, 35.45, 11.23, 25.29, 20.21]'; 
a3_9000 = 0.9803;
b3_9000 = 69.7255;
c3_9000 = 100.3024;
% rep percent thresh 10000:
y3_10000 = [96.78, 53.52, 24.51, 43.16, 38.77]';
a3_10000 = 0.9684;
b3_10000 = 21.6530;
c3_10000 = 100.7064;
% rep percent thresh 7000:
y3_7000 = [73.34, 8.59, 0.68, 2.34, 0.98]'; 
a3_7000 = 0.9957;
b3_7000 = 500.0000;
c3_7000 = 100.0000;

%% (4) stylegan2 CelebA:
x4 = [17.30, 21.34, 23.29]'; % ID
x4_norm = (x4-x_mean_common) / x_std_common;
xx4 = linspace(15,25,50); % for x = ID
% rep percent thresh 8000:
y4_8000 = [76.86, 21.19, 15.14]'; 
a4_8000 = 0.9747;
b4_8000 = 73.6460;
c4_8000 = 100.0000;
% rep percent thresh 9000:
y4_9000 = [91.41, 44.43, 33.01]';
a4_9000 = 0.9664;
b4_9000 = 32.2682;
c4_9000 = 101.8082;
% rep percent thresh 10000:
y4_10000 = [98.05, 67.77, 54.39]';
a4_10000 = 0.9607;
b4_10000 = 15.2072;
c4_10000 = 100.1992;
% rep percent thresh 7000:
y4_7000 = [54.00, 4.59, 3.91]';
a4_7000 = 0.9929;
b4_7000 = 491.6040;
c4_7000 = 100.0026;

%% (5) biggan LSUN:
x5 = [14.87, 20.80, 27.06, 29.57]'; % ID
x5_norm = (x5-x_mean_common) / x_std_common;
xx5 = linspace(10,35,50); % for x = ID
% rep percent thresh 8000:
y5_8000 = [38.57, 27.15, 0, 0.20]';
a5_8000 = 0.9756;
b5_8000 = 36.4029;
c5_8000 = 101.6608;
% rep percent thresh 9000:
y5_9000 = [66.41, 56.54, 0.39, 1.37]';
a5_9000 = 0.9687;
b5_9000 = 25.1107;
c5_9000 = 101.2578;
% rep percent thresh 10000:
y5_10000 = [88.57, 76.17, 4.10, 8.50]'; 
a5_10000 = 0.9649;
b5_10000 = 20.4582;
c5_10000 = 100.6128;
% rep percent thresh 7000:
y5_7000 = [18.95, 8.79, 0, 0]'; 
a5_7000 = 0.9856;
b5_7000 = 78.7101;
c5_7000 = 100.0000;

%% (6) stylegan2 LSUN:
x6 = [14.87, 20.80, 27.06, 29.57, 33.60]'; % ID
x6_norm = (x6-x_mean_common) / x_std_common;
xx6 = linspace(10,35,50); % for x = ID
% rep percent thresh 8000:
y6_8000 = [92.38, 27.93, 1.17, 3.03, 4.30]'; 
a6_8000 = 0.9735;  
b6_8000 = 51.5465;  
c6_8000 = 100.3704;
% rep percent thresh 9000:
y6_9000 = [96.09, 42.19, 4.69, 10.06, 12.01]'; 
a6_9000 = 0.9684; 
b6_9000 = 31.0002;
c6_9000 = 101.0800;
% rep percent thresh 10000:
y6_10000 = [98.14, 61.91, 13.87, 22.56, 22.36]';
a6_10000 = 0.9636;
b6_10000 = 17.7038;
c6_10000 = 100.6966;
% rep percent thresh 7000:
y6_7000 = [85.84, 15.53, 0.29, 0.98, 0.88]'; 
a6_7000 = 0.9805;
b6_7000 = 95.2465;
c6_7000 = 100.0000;


%%
x_all = {x1,x2,x3,x4,x5,x6};
x_norm_all = {x1_norm,x2_norm,x3_norm,x4_norm,x5_norm,x6_norm};
xx_all = {xx1,xx2,xx3,xx4,xx5,xx6};
% rep percent thresh 8000:
y_8000_all = {y1_8000,y2_8000,y3_8000,y4_8000,y5_8000,y6_8000};
a_8000_all = {a1_8000,a2_8000,a3_8000,a4_8000,a5_8000,a6_8000};
b_8000_all = {b1_8000,b2_8000,b3_8000,b4_8000,b5_8000,b6_8000};
c_8000_all = {c1_8000,c2_8000,c3_8000,c4_8000,c5_8000,c6_8000};
% rep percent thresh 9000:
y_9000_all = {y1_9000,y2_9000,y3_9000,y4_9000,y5_9000,y6_9000};
a_9000_all = {a1_9000,a2_9000,a3_9000,a4_9000,a5_9000,a6_9000};
b_9000_all = {b1_9000,b2_9000,b3_9000,b4_9000,b5_9000,b6_9000};
c_9000_all = {c1_9000,c2_9000,c3_9000,c4_9000,c5_9000,c6_9000};
% rep percent thresh 10000:
y_10000_all = {y1_10000,y2_10000,y3_10000,y4_10000,y5_10000,y6_10000};
a_10000_all = {a1_10000,a2_10000,a3_10000,a4_10000,a5_10000,a6_10000};
b_10000_all = {b1_10000,b2_10000,b3_10000,b4_10000,b5_10000,b6_10000};
c_10000_all = {c1_10000,c2_10000,c3_10000,c4_10000,c5_10000,c6_10000};
% rep percent thresh 7000:
y_7000_all = {y1_7000,y2_7000,y3_7000,y4_7000,y5_7000,y6_7000};
a_7000_all = {a1_7000,a2_7000,a3_7000,a4_7000,a5_7000,a6_7000};
b_7000_all = {b1_7000,b2_7000,b3_7000,b4_7000,b5_7000,b6_7000};
c_7000_all = {c1_7000,c2_7000,c3_7000,c4_7000,c5_7000,c6_7000};

title_list = {'biggan FLOWER', 'stylegan2 FLOWER', 'biggan CelebA',...
              'stylegan2 CelebA', 'biggan LSUN', 'stylegan2 LSUN'};

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
    plot(x_real,y,'.','color',rgb('RoyalBlue'),'MarkerSize',60);
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
    plot(x_real,y,'.','color',rgb('Turquoise'),'MarkerSize',60);
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
    plot(x_real,y,'.','color',rgb('YellowGreen'),'MarkerSize',60);
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
    
    lgd = legend(h, '\alpha=7000','\alpha=8000','\alpha=9000','\alpha=10000');
    lgd.FontSize = 15;
    hold off;


end

