# for NN query simCLR:
# --batch_size 1

python NNquery_simCLR_myTest.py --model_path results/CelebA_128/best_128_0.5_200_26_2000_model.pth \
 --dataset CelebA_128_sub1000 --data_dir /eecf/cbcsl/data100b/Chenqi/stylegan2/datasets_images/CelebA_128_sub1000/ \
 --gan_dir /eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NNquery_simCLR/CelebA_128_sub1000/fakes004933/view_sampleSheetImgs/ \
 --result_dir /eecf/cbcsl/data100b/Chenqi/stylegan2/imgs/NNquery_simCLR/CelebA_128_sub1000/fakes004933/ \
 --num_row 32 --num_col 32 --mean_std_data_dir /eecf/cbcsl/data100b/wen/Data/Datasets/celeba_cropped/ \
 --old_batch_size 26


  