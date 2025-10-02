# Difficult sample mining in knowledge distillation


## Environments:
- Python 3.8
- PyTorch 1.7.0
Install the package:
```
sudo pip3 install -r requirements.txt
sudo python setup.py develop
```

## datasets
* CIFAR-100 is a common dataset for image classification consisting of 50,000 training and 10,000 validation images. It contains 100 classes and its image size is 32×32.
* ImageNet is a large-scale dataset for image classification containing 1,000 categories and around 1.28 million training and 50,000 validation images.


## Training ON CIFAR-100
- Download the [`cifar_teachers.tar`](<https://github.com/megvii-research/mdistiller/releases/tag/checkpoints>) and untar it to `./download_ckpts` via `tar xvf cifar_teachers.tar`.


1. For KD
  ```bash
  # KD
  python tools/train.py --cfg configs/cifar100/kd/resnet32x4_resnet8x4.yaml
  # KD+Ours
  python tools/train.py --cfg configs/cifar100/kd/resnet32x4_resnet8x4.yaml --logit-stand --base-temp 2 --kd-weight 9 
  ```
2. For DKD
  ```bash
  # DKD
  python tools/train.py --cfg configs/cifar100/dkd/resnet32x4_resnet8x4.yaml 
  # DKD+Ours
  python tools/train.py --cfg configs/cifar100/dkd/resnet32x4_resnet8x4.yaml --logit-stand --base-temp 2 --kd-weight 9 
```
3. For MLKD
  ```bash
  # MLKD
  python tools/train.py --cfg configs/cifar100/mlkd/resnet32x4_resnet8x4.yaml
  # MLKD+Ours
  python tools/train.py --cfg configs/cifar100/mlkd/resnet32x4_resnet8x4.yaml --logit-stand --base-temp 2 --kd-weight 9 
```
4. For CTKD
Please refer to [CTKD](./CTKD).

5. For LKD
```bash  
  # LKD+Ours
  python tools/train.py --cfg configs/cifar100/kd/resnet32x4_resnet8x4.yaml --logit-stand --base-temp 2 --kd-weight 9 
```

# Training on ImageNet

- Download the dataset at <https://image-net.org/> and put it to `./data/imagenet`

```bash
  # KD
  python tools/train.py --cfg configs/imagenet/r34_r18/kd.yaml
  # LKD+Ours
  python tools/train.py --cfg configs/imagenet/r34_r18/kd.yaml --logit-stand --base-temp 2 --kd-weight 9 
```

