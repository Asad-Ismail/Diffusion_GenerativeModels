
## [![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/Asad-Ismail/Diffusion_GenerativeModels/issues)[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FAsad-Ismail%2FDiffusion_GenerativeModels&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

# Deep learning based image and Image/Text Generative models

# Denoising Diffusion Implicit Models
On Oxford Flower dataset
Image size 128 x 128
 
 From noise to final images in 200 steps
 
  <p align="center">
    <img src="images/flowers_gyf.gif" alt="pruning" />
  </p>
   <p align="center"> 
  
  Final Images @200 step
  
  <p align="center">
    <img src="images/flowers.png" alt="pruning",width="200" height="300"  />
  </p>
   <p align="center"> 

After training basic diffusion model on flower dataset we will now train the ImageGen diffusion model on small splash dataset with intermediate resolution
## ImageGen
After training ImageGen on unconditional small splash dataset, novel new images generated of intermediate resolution (512 x 512) by imagegen looks like below
 
 
 <p align="center">
  <img alt="Light" src="images/splash_results/1272.png" width="45%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Dark" src="images/splash_results/1357.png" width="45%">
</p>

 <p align="center">
  <img alt="Light" src="images/splash_results/991.png" width="45%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Dark" src="images/splash_results/486.png" width="45%">
</p>

 <p align="center">
  <img alt="Light" src="images/splash_results/570.png" width="45%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Dark" src="images/splash_results/576.png" width="45%">
</p>

 <p align="center">
  <img alt="Light" src="images/splash_results/622.png" width="45%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Dark" src="images/splash_results/659.png" width="45%">
</p>

  


### References
```
1) @article{Song, Jiaming, Chenlin Meng, and Stefano Ermon. "Denoising diffusion implicit models." arXiv preprint arXiv:2010.02502 (2020)}
