# StyleCLR

## Todo

- [x] SimCLR reimplement
  - [x] Trainer
    - [x] Training function (done by [Eugene](https://github.com/eugene87222))
    - [x] Loss (done by [Eugene](https://github.com/eugene87222))
      - Modified from [[leftthomas/SimCLR]](https://github.com/leftthomas/SimCLR)
    - [x] Continue Training mode (done by [Ethan](https://github.com/IShengFang))
  - [x] Backbone (done by [Eugene](https://github.com/eugene87222))
    - `torchvision.models`
    - Modified from [[sthalles/SimCLR]](https://github.com/sthalles/SimCLR)
  - [x] Augmentation  (done by [Eugene](https://github.com/eugene87222))
    - Newer version of `torchvision` can do the augmentation on tensor
    - Use [`kornia`](https://github.com/kornia/kornia)
    - Ref: [[sthalles/SimCLR]](https://github.com/sthalles/SimCLR)
  - [x] Evaluator (done by [Ethan](https://github.com/IShengFang))
    - Linear evaluation protocol
  - [x] Dataset
    - [x] `ContrastiveDataset` (done by [Eugene](https://github.com/eugene87222))
    - [x] ImageNet (downloaded)
      - [x] `ImageNetDataset` (done by [Eugene](https://github.com/eugene87222))
    - [x] ImageNet64 (downloaded)
      - [x] `ImageNet64Dataset` (done by [Ethan](https://github.com/IShengFang))
    - [x] [Tiny-Imagenet-200](https://github.com/rmccorm4/Tiny-Imagenet-200) (downloaded)
      - [x] `TinyImageNetDataset` (done by [Eugene](https://github.com/eugene87222))
    - [x] CIFAR-10 (`torchvision`) (done by [Ethan](https://github.com/IShengFang))
  - [x] Speedup
    - [x] Reduce image size from 224 to 64
    - [x] IO
      - [x] Can be solved by moving data to SSD
      - Downsampled version of ImageNet (32, 64)
        - https://patrykchrabaszcz.github.io/Imagenet32/
  - [x] Integration (done by [Ethan](https://github.com/IShengFang))

- [x] Style Transfer
  - [x] [AdaIN](https://github.com/naoto0804/pytorch-AdaIN)
    - [x] Pack to `nn.Module` (done by [Ethan](https://github.com/IShengFang))
      - AdaIN interface modified from [[IShengFang/Self-Contained Styleization]](https://github.com/IShengFang/Self-Contained_Stylization) and the model weights is from [[naoto0804/pytorch-AdaIN]](https://github.com/naoto0804/pytorch-AdaIN)
      - Plz be aware of the input and output range -> [0, 1]

- [ ] Experiment
  - [x] Baseline
    - [x] 64x64, bs=256, 100 epochs (trained by [Eugene](https://github.com/eugene87222))
  - [ ] Use style images
  - [x] Random style statistic
    - [x] 64x64, bs=256, 100 epochs, alpha=0.3 (trained by [Yolin](https://github.com/YU-LINs995503))
    - [x] 64x64, bs=256, 100 epochs, alpha=0.5 (trained by [Ethan](https://github.com/IShengFang))
    - [x] 64x64, bs=256, 100 epochs, alpha=0.7 (trained by [Ethan](https://github.com/IShengFang))
  - [x] CLR on TinyImageNet
    - [x] 64x64, bs=1024, 1000 epochs (trained by [Eugene](https://github.com/eugene87222))
    - [x] 64x64, bs=1024, 1000 epochs, alpha=0.3 (trained by [Ethan](https://github.com/IShengFang))
    - [x] 64x64, bs=1024, 1000 epochs, alpha=0.5 (trained by [Ethan](https://github.com/IShengFang))
    - [x] 64x64, bs=1024, 1000 epochs, alpha=0.5 (trained by [Ethan](https://github.com/IShengFang))

## Usage

## Results
