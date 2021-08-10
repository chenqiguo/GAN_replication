# for NN query simCLR:
# --batch_size 1

#python NNquery_simCLR_myTest.py --model_path results/FLOWER_128/best_128_0.5_200_26_2000_model.pth \
# --dataset FLOWER_128_train --data_dir /eecf/cbcsl/data100b/Chenqi/stylegan2/datasets_images/FLOWER_128/ \
# --gan_dir /eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NNquery_simCLR/FLOWER_128/fakes002526/view_sampleSheetImgs/ \
# --result_dir /eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NNquery_simCLR/FLOWER_128/fakes002526/
 
#python NNquery_simCLR_myTest.py --model_path results/FLOWER_128/best_128_0.5_200_26_2000_model.pth \
# --dataset FLOWER_128_sub1000 --data_dir /eecf/cbcsl/data100b/Chenqi/stylegan2/datasets_images/FLOWER_128_sub1000/ \
# --gan_dir /eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NNquery_simCLR/FLOWER_128_sub1000_resume/fakes003248/view_sampleSheetImgs/ \
# --result_dir /eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NNquery_simCLR/FLOWER_128_sub1000_resume/fakes003248/ \
# --num_row 32 --num_col 32 --mean_std_data_dir /eecf/cbcsl/data100b/Chenqi/data/flower/ \
# --old_batch_size 26

#python NNquery_simCLR_myTest.py --model_path results/FLOWER_128/best_128_0.5_200_26_2000_model.pth \
# --dataset FLOWER_128_sub4000 --data_dir /eecf/cbcsl/data100b/Chenqi/stylegan2/datasets_images/FLOWER_128_sub4000/ \
# --gan_dir /eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NNquery_simCLR/FLOWER_128_sub4000_resume/fakes003248/view_sampleSheetImgs/ \
# --result_dir /eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NNquery_simCLR/FLOWER_128_sub4000_resume/fakes003248/ \
# --num_row 32 --num_col 32 --mean_std_data_dir /eecf/cbcsl/data100b/Chenqi/data/flower/ \
# --old_batch_size 26


# for StyleGAN2:
python NNquery_simCLR_myTest.py --model_path results_v2/FLOWER_128/best_128_0.5_200_26_2000_model.pth \
 --dataset FLOWER_128_sub1000 --data_dir /eecf/cbcsl/data100b/Chenqi/stylegan2/datasets_images/FLOWER_128_sub1000/ \
 --gan_dir /eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NNquery_simCLR_v2/FLOWER_128_sub1000_resume/fakes003248/view_sampleSheetImgs/ \
 --result_dir /eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NNquery_simCLR_v2/FLOWER_128_sub1000_resume/fakes003248/ \
 --num_row 32 --num_col 32 --mean_std_data_dir /eecf/cbcsl/data100b/Chenqi/data/flower/ \
 --old_batch_size 26

python NNquery_simCLR_myTest.py --model_path results_v2/FLOWER_128/best_128_0.5_200_26_2000_model.pth \
 --dataset FLOWER_128_sub4000 --data_dir /eecf/cbcsl/data100b/Chenqi/stylegan2/datasets_images/FLOWER_128_sub4000/ \
 --gan_dir /eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NNquery_simCLR_v2/FLOWER_128_sub4000_resume/fakes003248/view_sampleSheetImgs/ \
 --result_dir /eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NNquery_simCLR_v2/FLOWER_128_sub4000_resume/fakes003248/ \
 --num_row 32 --num_col 32 --mean_std_data_dir /eecf/cbcsl/data100b/Chenqi/data/flower/ \
 --old_batch_size 26

python NNquery_simCLR_myTest.py --model_path results_v2/FLOWER_128/best_128_0.5_200_26_2000_model.pth \
 --dataset FLOWER_128_train --data_dir /eecf/cbcsl/data100b/Chenqi/stylegan2/datasets_images/FLOWER_128/ \
 --gan_dir /eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NNquery_simCLR_v2/FLOWER_128/fakes002526/view_sampleSheetImgs/ \
 --result_dir /eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NNquery_simCLR_v2/FLOWER_128/fakes002526/ \
 --num_row 32 --num_col 32 --mean_std_data_dir /eecf/cbcsl/data100b/Chenqi/data/flower/ \
 --old_batch_size 26


# for BigGAN:
python NNquery_simCLR_myTest.py --model_path results_v2/FLOWER_128/best_128_0.5_200_26_2000_model.pth \
 --dataset FLOWER_128_sub1000 --data_dir /eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/FLOWER_128_sub1000/ \
 --gan_dir /eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/biggan/NN_query/FLOWER_128_sub1000/Itr38950/view_sampleSheetImgs/ \
 --result_dir /eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/biggan/NNquery_simCLR_v2/FLOWER_128_sub1000/Itr38950/ \
 --num_row 32 --num_col 32 --mean_std_data_dir /eecf/cbcsl/data100b/Chenqi/data/flower/ \
 --old_batch_size 26






