# Look-Ahead Training with Learned Reflectance Loss for Single-Image SVBRDF Estimation

This is code of "Look-Ahead Training with Learned Reflectance Loss for Single-Image SVBRDF Estimation" [Project](https://people.engr.tamu.edu/nimak/Papers/SIGAsia2022_LookAhead/index.html) | [Paper](https://people.engr.tamu.edu/nimak/Papers/SIGAsia2022_LookAhead/final_paper.pdf)

To set up environment, please run this command below (tesed in Linux environment):

```
conda env create -f env.yml
```


Before running inference, please download:

1. our pretrained model from this [link](https://drive.google.com/file/d/1TvuTHtJQt8oTbLUiZ4_-mdORanDRlB-j/view?usp=share_link) 
2. scaled real dataset with MaterialGAN and our dataset with ground truth [link](https://drive.google.com/file/d/1eggbsN5adCxBgiSzPyBXL0jBWv8aOEdR/view?usp=share_link)

please save the download model to `./ckpt/' and extract data to './dataset':

To run inference on our and MaterialGAN's dataset with ground truth, please use this command:

```
python meta_test.py --fea all_N1 --wN_outer 80 --gamma --cuda --test_img $mode --name $name --val_step 7 --wR_outer 5 --loss_after1 TD --Wfea_vgg 5e-2 --Wdren_outer 10 --WTDren_outer 10 --adjust_light

```

where `$mode` set as `OurReal2` for our test dataset and `MGReal2` for MaterialGAN dataset, `$name` will be saved path

To run inference on real captured dataset without ground truth, please use this command:

```
python meta_test.py --val_root $path ---fea all_N1 --wN_outer 80 --gamma --cuda --test_img Real --name $name --val_step 7 --wR_outer 5 --loss_after1 TD --Wfea_vgg 5e-2 --Wdren_outer 10 --WTDren_outer 10 --adjust_light

```

where `$path` point to the directory of test real images, `$name` is the saved path

We also provide higher resolution version of unscaled real scenes: [link](https://drive.google.com/file/d/1kzJicyd9Dn-cGNWJCDqJ4fuh5b_NDajW/view?usp=share_link)

This code is not cleaned version, will clean it up soon. feel free to email me if you have any questions: 1992zhouxilong@gmail.com. Thanks for your understanding!