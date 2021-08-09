function [MAE_sizeRep, MAE_repSize, Rsqr_sizeRep, Rsqr_repSize] = computeMAE_wrapper_func(f_sizeID, x_mean_common, x_std_common, a_IDrep, b_IDrep, c_IDrep, size, rep, this_title)

% NOTE: I need to global normalize ID here!
f_sizeRep = a_IDrep.^(b_IDrep*(f_sizeID-x_mean_common)/x_std_common-c_IDrep);
f_repSize = finverse(f_sizeRep);
% compute MAE:
MAE_sizeRep = computeMAE_func(f_sizeRep, size, rep);
MAE_repSize = computeMAE_func(f_repSize, rep, size);
% compute R^2:
Rsqr_sizeRep = computeRsqr_func(f_sizeRep, size, rep);
Rsqr_repSize = computeRsqr_func(f_repSize, rep, size);

% plot f_sizeRep and f_repSize:
figure;
fplot(f_sizeRep, 'r-','LineWidth', 3);
ylim([0 100])
hold on
plot(size,rep,'b.','MarkerSize',60);
grid on;
xticks(floor(min(size)) : (ceil(max(size))-floor(min(size)))/5 : ceil(max(size))); % 1000
yticks(0 : 20 : 100);
title(this_title,'FontSize', 20);
xlabel('dataset size', 'FontSize', 18);
ylabel('replication percent', 'FontSize', 18');
ax = gca;
% Set x and y font sizes.
ax.XAxis.FontSize = 18;
ax.YAxis.FontSize = 18;

figure;
fplot(f_repSize, 'r-','LineWidth', 3);
xlim([0 100])
ylim([0 ceil(max(size))])
hold on
plot(rep,size,'b.','MarkerSize',60);
grid on;
yticks(floor(min(size)) : (ceil(max(size))-floor(min(size)))/5 : ceil(max(size)));
xticks(0 : 20 : 100);
title(this_title,'FontSize', 20);
ylabel('dataset size', 'FontSize', 18);
xlabel('replication percent', 'FontSize', 18');
ax = gca;
% Set x and y font sizes.
ax.XAxis.FontSize = 18;
ax.YAxis.FontSize = 18;

end