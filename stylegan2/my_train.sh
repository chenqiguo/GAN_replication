#python run_training.py \
# --num-gpus=1 --data-dir=datasets \
# --config=config-f --dataset=FLOWER_128_sub1000 \
# --total-kimg=25000 --gamma=100 \
# --result-dir=results/results_FLOWER_128_sub1000_resume \
# --resume-pkl=results/results_FLOWER_128_sub1000/00000-stylegan2-FLOWER_128_sub1000-1gpu-config-f/network-snapshot-002526.pkl


## FLOWER 256X256:

python run_training.py \
 --num-gpus=1 --data-dir=datasets \
 --config=config-f --dataset=FLOWER_256_sub1000 \
 --total-kimg=25000 --gamma=100 \
 --result-dir=results/results_FLOWER_256_sub1000 \
 --resume-pkl=results/results_FLOWER_256_sub1000/00001-stylegan2-FLOWER_256_sub1000-1gpu-config-f/network-snapshot-000322.pkl

python run_training.py \
 --num-gpus=1 --data-dir=datasets \
 --config=config-f --dataset=FLOWER_256_sub4000 \
 --total-kimg=25000 --gamma=100 \
 --result-dir=results/results_FLOWER_256_sub4000 \
 --resume-pkl=results/results_FLOWER_256_sub4000/00001-stylegan2-FLOWER_256_sub4000-1gpu-config-f/network-snapshot-000725.pkl

python run_training.py \
 --num-gpus=1 --data-dir=datasets \
 --config=config-f --dataset=FLOWER_256_sub6000 \
 --total-kimg=25000 --gamma=100 \
 --result-dir=results/results_FLOWER_256_sub6000 \
 --resume-pkl=results/results_FLOWER_256_sub6000/00001-stylegan2-FLOWER_256_sub6000-1gpu-config-f/network-snapshot-000725.pkl

python run_training.py \
 --num-gpus=1 --data-dir=datasets \
 --config=config-f --dataset=FLOWER_256 \
 --total-kimg=25000 --gamma=100 \
 --result-dir=results/results_FLOWER_256 \
 --resume-pkl=results/results_FLOWER_256/00001-stylegan2-FLOWER_256-1gpu-config-f/network-snapshot-000725.pkl


## CIFAR10:

python run_training.py \
 --num-gpus=1 --data-dir=datasets \
 --config=config-f --dataset=CIFAR10_32_sub1000 \
 --total-kimg=25000 --gamma=100 \
 --result-dir=results/results_CIFAR10_32_sub1000 \
 --resume-pkl=results/results_CIFAR10_32_sub1000/00000-stylegan2-CIFAR10_32_sub1000-1gpu-config-f/network-snapshot-003014.pkl

python run_training.py \
 --num-gpus=1 --data-dir=datasets \
 --config=config-f --dataset=CIFAR10_32_sub4000 \
 --total-kimg=25000 --gamma=100 \
 --result-dir=results/results_CIFAR10_32_sub4000 \
 --resume-pkl=results/results_CIFAR10_32_sub4000/00000-stylegan2-CIFAR10_32_sub4000-1gpu-config-f/network-snapshot-003014.pkl

python run_training.py \
 --num-gpus=1 --data-dir=datasets \
 --config=config-f --dataset=CIFAR10_32_sub8000 \
 --total-kimg=25000 --gamma=100 \
 --result-dir=results/results_CIFAR10_32_sub8000 \
 --resume-pkl=results/results_CIFAR10_32_sub8000/00000-stylegan2-CIFAR10_32_sub8000-1gpu-config-f/network-snapshot-003014.pkl

python run_training.py \
 --num-gpus=1 --data-dir=datasets \
 --config=config-f --dataset=CIFAR10_32_sub10000 \
 --total-kimg=25000 --gamma=100 \
 --result-dir=results/results_CIFAR10_32_sub10000 \
 --resume-pkl=results/results_CIFAR10_32_sub10000/00000-stylegan2-CIFAR10_32_sub10000-1gpu-config-f/network-snapshot-002009.pkl



#python run_training.py \
# --num-gpus=1 --data-dir=datasets \
# --config=config-f --dataset=CelebA_128_sub200 \
# --total-kimg=25000 --gamma=100 \
# --result-dir=results/results_CelebA_128_sub200 \

#python run_training.py \
# --num-gpus=1 --data-dir=datasets \
# --config=config-f --dataset=CelebA_128_sub600 \
# --total-kimg=25000 --gamma=100 \
# --result-dir=results/results_CelebA_128_sub600 \

# python run_training.py \
# --num-gpus=1 --data-dir=datasets \
# --config=config-f --dataset=CelebA_128_sub1000 \
# --total-kimg=25000 --gamma=100 \
# --result-dir=results/results_CelebA_128_sub1000 \

# python run_training.py \
# --num-gpus=1 --data-dir=datasets \
# --config=config-f --dataset=CelebA_128_sub4000 \
# --total-kimg=25000 --gamma=100 \
# --result-dir=results/results_CelebA_128_sub4000 \

#python run_training.py \
# --num-gpus=1 --data-dir=datasets \
# --config=config-f --dataset=CelebA_128_sub8000 \
# --total-kimg=25000 --gamma=100 \
# --result-dir=results/results_CelebA_128_sub8000 \

#python run_training.py \
# --num-gpus=1 --data-dir=datasets \
# --config=config-f --dataset=MNIST_128_sub10000 \
# --total-kimg=25000 --gamma=100 \
# --result-dir=results/results_MNIST_128_sub10000 \

#python run_training.py \
# --num-gpus=1 --data-dir=datasets \
# --config=config-f --dataset=MNIST_128_sub30000 \
# --total-kimg=25000 --gamma=100 \
# --result-dir=results/results_MNIST_128_sub30000 \

#python run_training.py \
# --num-gpus=1 --data-dir=datasets \
# --config=config-f --dataset=MNIST_128_train \
# --total-kimg=25000 --gamma=100 \
# --result-dir=results/results_MNIST_128_train \

#python run_training.py \
# --num-gpus=1 --data-dir=datasets \
# --config=config-f --dataset=LSUN_128_sub10000 \
# --total-kimg=25000 --gamma=100 \
# --result-dir=results/results_LSUN_128_sub10000 \

#python run_training.py \
# --num-gpus=1 --data-dir=datasets \
# --config=config-f --dataset=LSUN_128_sub30000 \
# --total-kimg=25000 --gamma=100 \
# --result-dir=results/results_LSUN_128_sub30000 \

#python run_training.py \
# --num-gpus=1 --data-dir=datasets \
# --config=config-f --dataset=LSUN_128_sub60000 \
# --total-kimg=25000 --gamma=100 \
# --result-dir=results/results_LSUN_128_sub60000 \

#python run_training.py \
# --num-gpus=1 --data-dir=datasets \
# --config=config-f --dataset=LSUN_128_sub1000 \
# --total-kimg=25000 --gamma=100 \
# --result-dir=results/results_LSUN_128_sub1000_resume \
# --resume-pkl=results/results_LSUN_128_sub1000/00000-stylegan2-LSUN_128_sub1000-1gpu-config-f/network-snapshot-001684.pkl

#python run_training.py \
# --num-gpus=1 --data-dir=datasets \
# --config=config-f --dataset=LSUN_128_sub5000 \
# --total-kimg=25000 --gamma=100 \
# --result-dir=results/results_LSUN_128_sub5000_resume \
# --resume-pkl=results/results_LSUN_128_sub5000/00000-stylegan2-LSUN_128_sub5000-1gpu-config-f/network-snapshot-001323.pkl

#python run_training.py \
# --num-gpus=1 --data-dir=datasets \
# --config=config-f --dataset=LSUN_128_sub200 \
# --total-kimg=25000 --gamma=100 \
# --result-dir=results/results_LSUN_128_sub200 \

