# Deepfake with Synthetic Face Image project

Please manually download the pre-trained pg-GAN model (provided by Nvidia), the trained feature extractor network, and the discovered feature axis from my personal dropbox link

Decompress the downloaded files and put it in project directory as the following format

root(d):
  asset_model(d):
    karras2018iclr-celebahq-1024x1024.pkl   # pretrained GAN from Nvidia
    cnn_face_attr_celeba(d):
      model_20180927_032934.h5              # trained feature extractor network
  asset_results(d):
    pg_gan_celeba_feature_direction_40(d):
      feature_direction_20181002_044444.pkl # feature axes




# References
TL-GAN: transparent latent-space GAN. SummitKwan. Retrieved from https://github.com/SummitKwan/transparent_latent_gan

FaceSwap. wuhuikai. Retrieved from https://github.com/wuhuikai/FaceSwap

