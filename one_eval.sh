export PRETRAINED=work_dirs/denoised_training/checkpoint_199.pth
export DATASET=noisy_mini-imagenet-gauss100-denoised
./run.sh imagenet_linear first_eval vit_small teacher 1
