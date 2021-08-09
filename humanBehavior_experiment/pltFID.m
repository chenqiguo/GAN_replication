% plot dataset size vs GAN FID
% for generated images.

%% parameters:
n_case = 6; % num of GAN curves --> NO MNIST !!!


%% (1) biggan FLOWER:
x1 = [1000, 2000, 4000, 6000, 8189]; % dataset size
xtick1 = 1000:1000:9000;
y1_FID = [70.9333, 153.1191, 142.0139, 140.1321, 78.4277]; % FID
y1_human = [3.44778, 2.35556, 2.44778, 2.27264, 2.58111]; % average rating
yconf1 = [3.58004,2.45782,2.53743,2.36002,2.69856,...
          2.46366,2.18525,2.35813,2.25329,3.31552]; % 95% confidence intervals

%% (2) stylegan2 FLOWER:
x2 = [1000, 4000, 8189]; % dataset size
xtick2 = 1000:1000:9000;
y2_FID = [125.1744, 78.2531, 33.4254]; % FID
y2_human = [2.76889, 3.18667, 3.55042]; % average rating
yconf2 = [2.87583,3.29398,3.65818,...
          3.44265,3.07935,2.66195]; % 95% confidence intervals

%% (3) biggan CelebA:
x3 = [200, 600, 1000, 4000, 8000]; % dataset size
xtick3 = 0:1000:8000;
y3_FID = [63.0902, 72.8129, 168.1227, 104.2261, 80.3904]; % FID
y3_human = [3.20333, 1.60306, 1.54889, 1.43, 1.55889]; % average rating
yconf3 = [3.35560,1.66461,1.61186,1.48088,1.60956,...
          1.50822,1.37912,1.48592,1.54150,3.05107]; % 95% confidence intervals

%% (4) stylegan2 CelebA:
x4 = [1000, 4000, 8000]; % dataset size
xtick4 = 1000:1000:8000;
y4_FID = [61.7308, 32.0846, 15.4268];  % FID
y4_human = [2.21, 2.44333, 3.51333];  % average rating
yconf4 = [2.28815,2.55114,3.64538,...
          3.38128,2.33553,2.13185]; % 95% confidence intervals

%% (5) biggan LSUN:
x5 = [200, 1000, 5000, 10000]; % dataset size
xtick5 = 0:1000:10000;
y5_FID = [147.4836, 177.6286, 133.9793, 109.8692]; % FID
y5_human = [2.72111, 1.98667, 2.46778, 2.25444]; % average rating
yconf5 = [2.83254,2.06798,2.55631,2.32742,...
          2.18147,2.37924,1.90536,2.60968]; % 95% confidence intervals

%% (6) stylegan2 LSUN:
x6 = [200, 1000, 5000, 10000, 30000]; % dataset size
xtick6 = 0:1000:30000;
y6_FID = [173.9122, 213.2400, 52.6651, 40.0893, 13.2056]; % FID
y6_human = [2.85556, 2.01444, 3.15111, 3.45111, 3.67778]; % average rating
yconf6 = [2.97855,2.09886,3.25666,3.56985,3.78774,...
          3.56781,3.33237,3.04556,1.93003,2.73256]; % 95% confidence intervals


%%
x_all = {x1,x2,x3,x4,x5,x6};
y_FID_all = {y1_FID,y2_FID,y3_FID,y4_FID,y5_FID,y6_FID};
y_human_all = {y1_human,y2_human,y3_human,y4_human,y5_human,y6_human};
yconf_all = {yconf1,yconf2,yconf3,yconf4,yconf5,yconf6};
title_list = {'biggan FLOWER', 'stylegan2 FLOWER', 'biggan CelebA',...
              'stylegan2 CelebA', 'biggan LSUN', 'stylegan2 LSUN'};

for i = 1:n_case % for each GAN curve
    x = x_all{i};
    y_FID = y_FID_all{i};
    y_human = y_human_all{i};
    this_title = title_list{i};
    
    xconf = [x x(end:-1:1)] ;         
    yconf = yconf_all{i};
    
    fig = figure;
    left_color = rgb('Crimson');
    right_color = rgb('DodgerBlue');
    set(fig,'defaultAxesColorOrder',[left_color; right_color]);

    yyaxis left
    title(this_title,'FontSize', 20);
    xlabel('dataset size', 'FontSize', 18);
    ylabel('generated images quality rating', 'FontSize', 18');
    
    yyaxis right
    ylabel('generated images FID', 'FontSize', 18');
    
    hold on
    
    yyaxis left
    p = fill(xconf,yconf,'red');
    p.FaceColor = [1 0.8 0.8];      
    p.EdgeColor = 'none';           
    hold on
    plot(x,y_human,'-','LineWidth', 2);
    hold on
    plot(x,y_human,'.','MarkerSize',30);
    ylim([0 5])
    
    yyaxis right
    plot(x,y_FID,'-','LineWidth', 2);
    hold on
    plot(x,y_FID,'.','MarkerSize',30);
    %ylim([floor(min(y_FID)) ceil(max(y_FID))])
    
    grid on;
    xlim([floor(min(x)) ceil(max(x))])
    %xticks(floor(min(x)) : (ceil(max(x))-floor(min(x)))/5 : ceil(max(x))); % 1000
    %yticks(0 : 1 : 5);
    
    
    ax = gca;
    % Set x and y font sizes.
    ax.XAxis.FontSize = 18;
    ax.YAxis(1).FontSize = 18;
    ax.Clipping = 'off';    % turn clipping off
    ax.YAxis(2).FontSize = 18;
    hold off

end



