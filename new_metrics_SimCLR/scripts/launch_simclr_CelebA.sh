# for training simCLR:
# --batch_size 1024
#python main.py --batch_size 320 --epochs 1000 
python main_chenqi.py --batch_size 26 --epochs 2000 \
 --dataset CelebA_128 --data_dir /eecf/cbcsl/data100b/wen/Data/Datasets/celeba_cropped/ \
 --label_file /eecf/cbcsl/data100b/wen/Data/Datasets/celeba/Anno/identity_CelebA.txt


# for linear evaluation:
#python linear.py --batch_size 1024 --epochs 200 \
#--model_path 'results/128_0.5_200_320_1000_model.pth'
  
  