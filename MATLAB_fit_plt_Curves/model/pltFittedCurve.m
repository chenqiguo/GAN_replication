% use this code to plot the curve fitted from fitExpModel_v2.m
% NOTE: since we do the normalization for x, we need to deal with it when
% plotting the curve here!


%% Finally, plot curves:
% for LOOCV:
%title_list = {'biggan FLOWER LOOCV', 'stylegan2 FLOWER LOOCV', 'biggan CelebA LOOCV',...
%              'stylegan2 CelebA LOOCV', 'biggan LSUN LOOCV', 'stylegan2 LSUN LOOCV'};
% for fit:
title_list = {'biggan FLOWER', 'stylegan2 FLOWER', 'biggan CelebA',...
              'stylegan2 CelebA', 'biggan LSUN', 'stylegan2 LSUN'};

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







