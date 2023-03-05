# Look-Ahead Training with Learned Reflectance Loss for Single-Image SVBRDF Estimation

This is code of "Look-Ahead Training with Learned Reflectance Loss for Single-Image SVBRDF Estimation" [Project](https://people.engr.tamu.edu/nimak/Papers/SIGAsia2022_LookAhead/index.html) | [Paper](https://people.engr.tamu.edu/nimak/Papers/SIGAsia2022_LookAhead/final_paper.pdf)

<img src='img/teaser.jpg'>

## Set up environment

To set up environment, please run this command below (tesed in Linux environment):

```
conda env create -f env.yml
```
## Inference

Before running inference, please download:

1. our pretrained model from this [link](https://drive.google.com/file/d/1TvuTHtJQt8oTbLUiZ4_-mdORanDRlB-j/view?usp=share_link) 
2. centeralized MaterialGAN and our dataset with ground truth [link](https://drive.google.com/file/d/1eggbsN5adCxBgiSzPyBXL0jBWv8aOEdR/view?usp=share_link)

please save the download model to `./ckpt/` and extract data to `./dataset`:



# To run inference on our and MaterialGAN's dataset with ground truth, please use this command:

```
python meta_test.py --fea all_N1 --wN_outer 80 --gamma --cuda --test_img $mode --name $name --val_step 7 --wR_outer 5 --loss_after1 TD --Wfea_vgg 5e-2 --Wdren_outer 10 --WTDren_outer 10 --adjust_light

```

where `$mode` set as `OurReal2` for our test dataset and `MGReal2` for MaterialGAN dataset, `$name` represents the saved path. Inside the saved path, `RenLPIPS` and `RenRMSE` are the lpips and rmse value. 

Here are some clarifications of saved images for each scene: `fea`: final SVBRDF, `fea0`: SVBRDF at step 0, `render_#`: rendered image under 8 test lightings, `render_o0`: rendered image under input lightings, `render_t0`: the input image, `progressive_img`: optimization process at step 0,1,2,5,7 (row 1-5)



To run inference on real captured dataset without ground truth, please first centeralized the specular highlight of input image and then run this command:

```
python meta_test.py --val_root $path --fea all_N1 --wN_outer 80 --gamma --cuda --test_img Real --name $name --val_step 7 --wR_outer 5 --loss_after1 TD --Wfea_vgg 5e-2 --Wdren_outer 10 --WTDren_outer 10 --adjust_light

```

where `$path` point to the directory of test real images, `$name` represents the saved path. Inside the saved path, the final feature maps are saved to `$name\fea` and optimization process at step 0,1,2,5,7 are saved to `$name\pro`

## Our Dataset

We also provide higher resolution version of unscaled real scenes: [link](https://drive.google.com/file/d/1kzJicyd9Dn-cGNWJCDqJ4fuh5b_NDajW/view?usp=share_link)

## Citation

If you find this work useful for your research, please cite:

```
@article{zhou2022look,
  title={Look-Ahead Training with Learned Reflectance Loss for Single-Image SVBRDF Estimation},
  author={Zhou, Xilong and Kalantari, Nima Khademi},
  journal={ACM Transactions on Graphics (TOG)},
  volume={41},
  number={6},
  pages={1--12},
  year={2022},
  publisher={ACM New York, NY, USA}
}

```

## Contact

This code is not clean version, will clean it up soon. feel free to email me if you have any questions: 1992zhouxilong@gmail.com. Thanks for your understanding!