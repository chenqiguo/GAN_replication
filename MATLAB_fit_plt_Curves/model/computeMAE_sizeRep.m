% calculate the median absolute error (MAE) for the size-replication curve,
% as well as the replication-size curve:
% since we have both size->id and id->replication,
% we can do then a change of variable to get size->replication


%% parameters:
n_case = 6; % num of GAN curves

x_tmp = [22.02, 24.70, 27.41, 28.99, 30.34,...
         11.90, 15.97, 17.30, 21.34, 23.29,...
         14.87, 20.80, 27.06, 29.57, 33.60];
x_mean_common = mean(x_tmp);
x_std_common = std(x_tmp);

% for LOOCV:
title_LOOCV_list = {'biggan FLOWER LOOCV', 'stylegan2 FLOWER LOOCV', 'biggan CelebA LOOCV',...
                    'stylegan2 CelebA LOOCV', 'biggan LSUN LOOCV', 'stylegan2 LSUN LOOCV'};
% for fit:
title_fit_list = {'biggan FLOWER', 'stylegan2 FLOWER', 'biggan CelebA',...
                  'stylegan2 CelebA', 'biggan LSUN', 'stylegan2 LSUN'};


%% (1) biggan FLOWER:
ID1 = [22.02, 24.70, 27.41, 28.99, 30.34]';
size1 = [1000, 2000, 4000, 6000, 8189]';
rep1 = [76.07, 34.77, 1.86, 4.49, 0.29].';
% the id->size func: val(x) = a*exp(b*x)
a_IDsize_1 = 4.485;
b_IDsize_1 = 0.2478;
% (a) fit_result: val(x) = a.^(b*x-c)
a_fit_IDrep_1 = 0.9621;
b_fit_IDrep_1 = 62.9262;
c_fit_IDrep_1 = 100.0000;
% (b) LOOCV_result 1pair:
a_LOOCV_IDrep_1 = 0.9766;
b_LOOCV_IDrep_1 = 415.9661;
c_LOOCV_IDrep_1 = 100.4215;

%% (2) stylegan2 FLOWER:
ID2 = [22.02, 27.41, 30.34]';
size2 = [1000, 4000, 8189]';
rep2 = [31.93, 1.76, 1.07]';
% the id->size func: val(x) = a*exp(b*x)
a_IDsize_2 = a_IDsize_1;
b_IDsize_2 = b_IDsize_1;
% (a) fit_result: val(x) = a.^(b*x-c)
a_fit_IDrep_2 = 0.9723;
b_fit_IDrep_2 = 116.3858;
c_fit_IDrep_2 = 100.0763;
% (b) LOOCV_result 1pair:
a_LOOCV_IDrep_2 = 0.9746;
b_LOOCV_IDrep_2 = 171.3563;
c_LOOCV_IDrep_2 = 100.4062;

%% (3) biggan CelebA:
ID3 = [11.90, 15.97, 17.30, 21.34, 23.29]';
size3 = [200, 600, 1000, 4000, 8000]';
rep3 = [84.18, 19.82, 4.30, 11.43, 6.05]';
% the id->size func: val(x) = a*exp(b*x)
a_IDsize_3 = 2.224;
b_IDsize_3 = 0.3515;
% (a) fit_result: val(x) = a.^(b*x-c)
a_fit_IDrep_3 = 0.9869;
b_fit_IDrep_3 = 130.0000;
c_fit_IDrep_3 = 100.0000;
% (b) LOOCV_result 1pair:
a_LOOCV_IDrep_3 = 0.9716;
b_LOOCV_IDrep_3 = 29.8608;
c_LOOCV_IDrep_3 = 100.4215;

%% (4) stylegan2 CelebA:
ID4 = [17.30, 21.34, 23.29]';
size4 = [1000, 4000, 8000]';
rep4 = [76.86, 21.19, 15.14]';
% the id->size func: val(x) = a*exp(b*x)
a_IDsize_4 = a_IDsize_3;
b_IDsize_4 = b_IDsize_3;
% (a) fit_result: val(x) = a.^(b*x-c)
a_fit_IDrep_4 = 0.9747;
b_fit_IDrep_4 = 73.6460;
c_fit_IDrep_4 = 100.0000;
% (b) LOOCV_result 1pair:
a_LOOCV_IDrep_4 = 0.9741;
b_LOOCV_IDrep_4 = 68.7727;
c_LOOCV_IDrep_4 = 100.4215;

%% (5) biggan LSUN:
ID5 = [14.87, 20.80, 27.06, 29.57]';
size5 = [200, 1000, 5000, 10000]';
rep5 = [38.57, 27.15, 0, 0.20]';
% the id->size func: val(x) = a*exp(b*x)
a_IDsize_5 = 3.141;
b_IDsize_5 = 0.2727;
% (a) fit_result: val(x) = a.^(b*x-c)
a_fit_IDrep_5 = 0.9756;
b_fit_IDrep_5 = 36.4029;
c_fit_IDrep_5 = 101.6608;
% (b) LOOCV_result 1pair:
a_LOOCV_IDrep_5 = 0.9739;
b_LOOCV_IDrep_5 = 28.6599;
c_LOOCV_IDrep_5 = 100.0893;

%% (6) stylegan2 LSUN:
ID6 = [14.87, 20.80, 27.06, 29.57, 33.60]';
size6 = [200, 1000, 5000, 10000, 30000]';
rep6 = [92.38, 27.93, 1.17, 3.03, 4.30]';
% the id->size func: val(x) = a*exp(b*x)
a_IDsize_6 = a_IDsize_5;
b_IDsize_6 = b_IDsize_5;
% (a) fit_result: val(x) = a.^(b*x-c)
a_fit_IDrep_6 = 0.9735;
b_fit_IDrep_6 = 51.5465;
c_fit_IDrep_6 = 100.3704;
% (b) LOOCV_result 1pair:
a_LOOCV_IDrep_6 = 0.9743;
b_LOOCV_IDrep_6 = 55.4611;
c_LOOCV_IDrep_6 = 100.3474;

%% LOOP through each case:

ID_all = {ID1,ID2,ID3,ID4,ID5,ID6};
size_all = {size1,size2,size3,size4,size5,size6};
rep_all = {rep1,rep2,rep3,rep4,rep5,rep6};
a_IDsize_all = {a_IDsize_1,a_IDsize_2,a_IDsize_3,a_IDsize_4,a_IDsize_5,a_IDsize_6};
b_IDsize_all = {b_IDsize_1,b_IDsize_2,b_IDsize_3,b_IDsize_4,b_IDsize_5,b_IDsize_6};
a_fit_IDrep_all = {a_fit_IDrep_1,a_fit_IDrep_2,a_fit_IDrep_3,a_fit_IDrep_4,a_fit_IDrep_5,a_fit_IDrep_6};
b_fit_IDrep_all = {b_fit_IDrep_1,b_fit_IDrep_2,b_fit_IDrep_3,b_fit_IDrep_4,b_fit_IDrep_5,b_fit_IDrep_6};
c_fit_IDrep_all = {c_fit_IDrep_1,c_fit_IDrep_2,c_fit_IDrep_3,c_fit_IDrep_4,c_fit_IDrep_5,c_fit_IDrep_6};
a_LOOCV_IDrep_all = {a_LOOCV_IDrep_1,a_LOOCV_IDrep_2,a_LOOCV_IDrep_3,a_LOOCV_IDrep_4,a_LOOCV_IDrep_5,a_LOOCV_IDrep_6};
b_LOOCV_IDrep_all = {b_LOOCV_IDrep_1,b_LOOCV_IDrep_2,b_LOOCV_IDrep_3,b_LOOCV_IDrep_4,b_LOOCV_IDrep_5,b_LOOCV_IDrep_6};
c_LOOCV_IDrep_all = {c_LOOCV_IDrep_1,c_LOOCV_IDrep_2,c_LOOCV_IDrep_3,c_LOOCV_IDrep_4,c_LOOCV_IDrep_5,c_LOOCV_IDrep_6};

% init mat to store computed MAE values:
MAE_sizeRep_fit_all = zeros(1,n_case);
MAE_repSize_fit_all = zeros(1,n_case);
MAE_sizeRep_LOOCV_all = zeros(1,n_case);
MAE_repSize_LOOCV_all = zeros(1,n_case);
% newly added: also compute the R^2 values:
Rsqr_sizeRep_fit_all = zeros(1,n_case);
Rsqr_repSize_fit_all = zeros(1,n_case);
Rsqr_sizeRep_LOOCV_all = zeros(1,n_case);
Rsqr_repSize_LOOCV_all = zeros(1,n_case);

for i = 1:n_case % for each GAN curve
    ID = ID_all{i};
    size = size_all{i};
    rep = rep_all{i};
    a_IDsize = a_IDsize_all{i};
    b_IDsize = b_IDsize_all{i};
    a_fit_IDrep = a_fit_IDrep_all{i};
    b_fit_IDrep = b_fit_IDrep_all{i};
    c_fit_IDrep = c_fit_IDrep_all{i};
    a_LOOCV_IDrep = a_LOOCV_IDrep_all{i};
    b_LOOCV_IDrep = b_LOOCV_IDrep_all{i};
    c_LOOCV_IDrep = c_LOOCV_IDrep_all{i};
    this_title_fit = title_fit_list{i};
    this_title_LOOCV = title_LOOCV_list{i};
    
    % the id->size func:
    syms x
    f_IDsize = a_IDsize*exp(b_IDsize*x);
    % the size->id func:
    f_sizeID = finverse(f_IDsize);
    
    % (a) fit_result:
    [MAE_sizeRep_fit, MAE_repSize_fit, Rsqr_sizeRep_fit, Rsqr_repSize_fit] = computeMAE_wrapper_func(f_sizeID, x_mean_common, x_std_common, a_fit_IDrep, b_fit_IDrep, c_fit_IDrep, size, rep, this_title_fit);
    
    % (b) LOOCV_result 1pair:
    [MAE_sizeRep_LOOCV, MAE_repSize_LOOCV, Rsqr_sizeRep_LOOCV, Rsqr_repSize_LOOCV] = computeMAE_wrapper_func(f_sizeID, x_mean_common, x_std_common, a_LOOCV_IDrep, b_LOOCV_IDrep, c_LOOCV_IDrep, size, rep, this_title_LOOCV);
    
    MAE_sizeRep_fit_all(i) = MAE_sizeRep_fit;
    MAE_repSize_fit_all(i) = MAE_repSize_fit;
    MAE_sizeRep_LOOCV_all(i) = MAE_sizeRep_LOOCV;
    MAE_repSize_LOOCV_all(i) = MAE_repSize_LOOCV;
    
    Rsqr_sizeRep_fit_all(i) = Rsqr_sizeRep_fit;
    Rsqr_repSize_fit_all(i) = Rsqr_repSize_fit;
    Rsqr_sizeRep_LOOCV_all(i) = Rsqr_sizeRep_LOOCV;
    Rsqr_repSize_LOOCV_all(i) = Rsqr_repSize_LOOCV;
    
end

















