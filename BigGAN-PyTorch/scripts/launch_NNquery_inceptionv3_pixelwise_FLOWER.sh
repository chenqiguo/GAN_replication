# for NN query simCLR:
# --batch_size 1


# for StyleGAN2: NOT modified yet!!!
python NNquery_inceptionv3_pixelwise_myTest.py \
 --dataset FLOWER_128_sub1000 --data_dir /eecf/cbcsl/data100b/Chenqi/stylegan2/datasets_images/FLOWER_128_sub1000/ \
 --gan_dir /eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NNquery_inception_v3/FLOWER_128_sub1000_resume/fakes003248/view_sampleSheetImgs/ \
 --result_dir /eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/stylegan2/NNquery_inception_v3/FLOWER_128_sub1000_resume/fakes003248/ \
 --num_row 32 --num_col 32 


# for BigGAN:
python NNquery_inceptionv3_pixelwise_myTest.py \
 --dataset FLOWER_128_sub1000 --data_dir /eecf/cbcsl/data100b/Chenqi/BigGAN-PyTorch/imgs/FLOWER_128_sub1000/ \
 --gan_dir /eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/biggan/NNquery_inceptionv3_pixelwise/FLOWER_128_sub1000/Itr38950/view_sampleSheetImgs/ \
 --result_dir /eecf/cbcsl/data100b/Chenqi/gan_results_for_presentation/biggan/NNquery_inceptionv3_pixelwise/FLOWER_128_sub1000/Itr38950/ \
 --num_row 32 --num_col 32 






