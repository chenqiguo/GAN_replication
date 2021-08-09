% calculate the median absolute error (MAE) for the ID-replication curve,
% for full set fit, LOOCV 1-shot and LOOCV 2-shot.

%% parameters:
n_case = 6; % num of GAN curves
% an array of all the dataset ID values:
x_tmp = [22.02, 24.70, 27.41, 28.99, 30.34,...
         11.90, 15.97, 17.30, 21.34, 23.29,...
         14.87, 20.80, 27.06, 29.57, 33.60];
x_mean_common = mean(x_tmp);
x_std_common = std(x_tmp);

%% (1) biggan FLOWER:
ID1 = [22.02, 24.70, 27.41, 28.99, 30.34]';
rep1 = [76.07, 34.77, 1.86, 4.49, 0.29].';
%{
% (a) fit_result: val(x) = a.^(b*x-c) for thresh 8000
a_fit_IDrep_1 = 0.9621;
b_fit_IDrep_1 = 62.9262;
c_fit_IDrep_1 = 100.0000;
%}
% (a) fit_result: val(x) = a.^(b*x-c) for thresh 9000
a_fit_IDrep_1 = 0.9595;
b_fit_IDrep_1 = 43.1907;
c_fit_IDrep_1 = 100.0000;
% (b) LOOCV_result 1pair:
a_LOOCV_IDrep_1p_1 = 0.9766;
b_LOOCV_IDrep_1p_1 = 415.9661;
c_LOOCV_IDrep_1p_1 = 100.4215;
% (c) LOOCV_result 2pairs:
a_LOOCV_IDrep_2p_1 = 0.9766;
b_LOOCV_IDrep_2p_1 = 414.2778;
c_LOOCV_IDrep_2p_1 = 100.4215;

%% (2) stylegan2 FLOWER:
ID2 = [22.02, 27.41, 30.34]';
rep2 = [31.93, 1.76, 1.07]';
%{
% (a) fit_result: val(x) = a.^(b*x-c) for thresh 8000
a_fit_IDrep_2 = 0.9723;
b_fit_IDrep_2 = 116.3858;
c_fit_IDrep_2 = 100.0763;
%}
% (a) fit_result: val(x) = a.^(b*x-c) for thresh 9000
a_fit_IDrep_2 = 0.9663;
b_fit_IDrep_2 = 61.3039;
c_fit_IDrep_2 = 100.6188;
% (b) LOOCV_result 1pair:
a_LOOCV_IDrep_1p_2 = 0.9746;
b_LOOCV_IDrep_1p_2 = 171.3563;
c_LOOCV_IDrep_1p_2 = 100.4062;
% (c) LOOCV_result 2pairs:
a_LOOCV_IDrep_2p_2 = 0.9746;
b_LOOCV_IDrep_2p_2 = 170.8761;
c_LOOCV_IDrep_2p_2 = 100.4062;

%% (3) biggan CelebA:
ID3 = [11.90, 15.97, 17.30, 21.34, 23.29]';
rep3 = [84.18, 19.82, 4.30, 11.43, 6.05]';
%{
% (a) fit_result: val(x) = a.^(b*x-c) for thresh 8000
a_fit_IDrep_3 = 0.9869;
b_fit_IDrep_3 = 130.0000;
c_fit_IDrep_3 = 100.0000;
%}
% (a) fit_result: val(x) = a.^(b*x-c) for thresh 9000
a_fit_IDrep_3 = 0.9803;
b_fit_IDrep_3 = 69.7255;
c_fit_IDrep_3 = 100.3024;
% (b) LOOCV_result 1pair:
a_LOOCV_IDrep_1p_3 = 0.9716;
b_LOOCV_IDrep_1p_3 = 29.8608;
c_LOOCV_IDrep_1p_3 = 100.4215;
% (c) LOOCV_result 2pairs:
a_LOOCV_IDrep_2p_3 = 0.9716;
b_LOOCV_IDrep_2p_3 = 27.4999;
c_LOOCV_IDrep_2p_3 = 100.4215;

%% (4) stylegan2 CelebA:
ID4 = [17.30, 21.34, 23.29]';
rep4 = [76.86, 21.19, 15.14]';
%{
% (a) fit_result: val(x) = a.^(b*x-c) for thresh 8000
a_fit_IDrep_4 = 0.9747;
b_fit_IDrep_4 = 73.6460;
c_fit_IDrep_4 = 100.0000;
%}
% (a) fit_result: val(x) = a.^(b*x-c) for thresh 9000
a_fit_IDrep_4 = 0.9664;
b_fit_IDrep_4 = 32.2682;
c_fit_IDrep_4 = 101.8082;
% (b) LOOCV_result 1pair:
a_LOOCV_IDrep_1p_4 = 0.9741;
b_LOOCV_IDrep_1p_4 = 68.7727;
c_LOOCV_IDrep_1p_4 = 100.4215;
% (c) LOOCV_result 2pairs:
a_LOOCV_IDrep_2p_4 = 0.9741;
b_LOOCV_IDrep_2p_4 = 68.6061;
c_LOOCV_IDrep_2p_4 = 100.4215;

%% (5) biggan LSUN:
ID5 = [14.87, 20.80, 27.06, 29.57]';
rep5 = [38.57, 27.15, 0, 0.20]';
%{
% (a) fit_result: val(x) = a.^(b*x-c) for thresh 8000
a_fit_IDrep_5 = 0.9756;
b_fit_IDrep_5 = 36.4029;
c_fit_IDrep_5 = 101.6608;
%}
% (a) fit_result: val(x) = a.^(b*x-c) for thresh 9000
a_fit_IDrep_5 = 0.9687;
b_fit_IDrep_5 = 25.1107;
c_fit_IDrep_5 = 101.2578;
% (b) LOOCV_result 1pair:
a_LOOCV_IDrep_1p_5 = 0.9739;
b_LOOCV_IDrep_1p_5 = 28.6599;
c_LOOCV_IDrep_1p_5 = 100.0893;
% (c) LOOCV_result 2pairs:
a_LOOCV_IDrep_2p_5 = 0.9739;
b_LOOCV_IDrep_2p_5 = 29.4884;
c_LOOCV_IDrep_2p_5 = 100.0893;

%% (6) stylegan2 LSUN:
ID6 = [14.87, 20.80, 27.06, 29.57, 33.60]';
rep6 = [92.38, 27.93, 1.17, 3.03, 4.30]';
%{
% (a) fit_result: val(x) = a.^(b*x-c) for thresh 8000
a_fit_IDrep_6 = 0.9735;
b_fit_IDrep_6 = 51.5465;
c_fit_IDrep_6 = 100.3704;
%}
% (a) fit_result: val(x) = a.^(b*x-c) for thresh 9000
a_fit_IDrep_6 = 0.9684;
b_fit_IDrep_6 = 31.0002;
c_fit_IDrep_6 = 101.0800;
% (b) LOOCV_result 1pair:
a_LOOCV_IDrep_1p_6 = 0.9743;
b_LOOCV_IDrep_1p_6 = 55.4611;
c_LOOCV_IDrep_1p_6 = 100.3474;
% (c) LOOCV_result 2pairs:
a_LOOCV_IDrep_2p_6 = 0.9743;
b_LOOCV_IDrep_2p_6 = 55.5556;
c_LOOCV_IDrep_2p_6 = 100.3474;


%% LOOP through each case:

ID_all = {ID1,ID2,ID3,ID4,ID5,ID6};
rep_all = {rep1,rep2,rep3,rep4,rep5,rep6};
a_fit_IDrep_all = {a_fit_IDrep_1,a_fit_IDrep_2,a_fit_IDrep_3,a_fit_IDrep_4,a_fit_IDrep_5,a_fit_IDrep_6};
b_fit_IDrep_all = {b_fit_IDrep_1,b_fit_IDrep_2,b_fit_IDrep_3,b_fit_IDrep_4,b_fit_IDrep_5,b_fit_IDrep_6};
c_fit_IDrep_all = {c_fit_IDrep_1,c_fit_IDrep_2,c_fit_IDrep_3,c_fit_IDrep_4,c_fit_IDrep_5,c_fit_IDrep_6};
a_LOOCV_IDrep_1p_all = {a_LOOCV_IDrep_1p_1,a_LOOCV_IDrep_1p_2,a_LOOCV_IDrep_1p_3,a_LOOCV_IDrep_1p_4,a_LOOCV_IDrep_1p_5,a_LOOCV_IDrep_1p_6};
b_LOOCV_IDrep_1p_all = {b_LOOCV_IDrep_1p_1,b_LOOCV_IDrep_1p_2,b_LOOCV_IDrep_1p_3,b_LOOCV_IDrep_1p_4,b_LOOCV_IDrep_1p_5,b_LOOCV_IDrep_1p_6};
c_LOOCV_IDrep_1p_all = {c_LOOCV_IDrep_1p_1,c_LOOCV_IDrep_1p_2,c_LOOCV_IDrep_1p_3,c_LOOCV_IDrep_1p_4,c_LOOCV_IDrep_1p_5,c_LOOCV_IDrep_1p_6};
a_LOOCV_IDrep_2p_all = {a_LOOCV_IDrep_2p_1,a_LOOCV_IDrep_2p_2,a_LOOCV_IDrep_2p_3,a_LOOCV_IDrep_2p_4,a_LOOCV_IDrep_2p_5,a_LOOCV_IDrep_2p_6};
b_LOOCV_IDrep_2p_all = {b_LOOCV_IDrep_2p_1,b_LOOCV_IDrep_2p_2,b_LOOCV_IDrep_2p_3,b_LOOCV_IDrep_2p_4,b_LOOCV_IDrep_2p_5,b_LOOCV_IDrep_2p_6};
c_LOOCV_IDrep_2p_all = {c_LOOCV_IDrep_2p_1,c_LOOCV_IDrep_2p_2,c_LOOCV_IDrep_2p_3,c_LOOCV_IDrep_2p_4,c_LOOCV_IDrep_2p_5,c_LOOCV_IDrep_2p_6};

% init mat to store computed MAE values:
MAE_fit_all = zeros(1,n_case);
MAE_LOOCV_1p_all = zeros(1,n_case);
MAE_LOOCV_2p_all = zeros(1,n_case);

for i = 1:n_case % for each GAN curve
    ID = ID_all{i};
    ID_norm = (ID - x_mean_common) / x_std_common;
    
    rep = rep_all{i};
    a_fit_IDrep = a_fit_IDrep_all{i};
    b_fit_IDrep = b_fit_IDrep_all{i};
    c_fit_IDrep = c_fit_IDrep_all{i};
    a_LOOCV_IDrep_1p = a_LOOCV_IDrep_1p_all{i};
    b_LOOCV_IDrep_1p = b_LOOCV_IDrep_1p_all{i};
    c_LOOCV_IDrep_1p = c_LOOCV_IDrep_1p_all{i};
    a_LOOCV_IDrep_2p = a_LOOCV_IDrep_2p_all{i};
    b_LOOCV_IDrep_2p = b_LOOCV_IDrep_2p_all{i};
    c_LOOCV_IDrep_2p = c_LOOCV_IDrep_2p_all{i};
    
    % the id->replication func:
    % (1) full set fit:
    syms x
    f_fit_IDrep = a_fit_IDrep^(b_fit_IDrep*x-c_fit_IDrep);
    MAE_fit_IDrep = computeMAE_func(f_fit_IDrep, ID_norm, rep);
    % (2) LOOCV 1-shot:
    syms x
    f_LOOCV_IDrep_1p = a_LOOCV_IDrep_1p^(b_LOOCV_IDrep_1p*x-c_LOOCV_IDrep_1p);
    MAE_LOOCV_IDrep_1p = computeMAE_func(f_LOOCV_IDrep_1p, ID_norm, rep);
    % (3) LOOCV 2-shot:
    syms x
    f_LOOCV_IDrep_2p = a_LOOCV_IDrep_2p^(b_LOOCV_IDrep_2p*x-c_LOOCV_IDrep_2p);
    MAE_LOOCV_IDrep_2p = computeMAE_func(f_LOOCV_IDrep_2p, ID_norm, rep);
    
    MAE_fit_all(i) = MAE_fit_IDrep;
    MAE_LOOCV_1p_all(i) = MAE_LOOCV_IDrep_1p;
    MAE_LOOCV_2p_all(i) = MAE_LOOCV_IDrep_2p;
    
end


