from trainer3 import Trainer

# from tester import Tester
from data_loader import Data_Loader
from torch.backends import cudnn

import torch

##### Import libary for dataloader #####
from Dataloader.transform.spatial_transforms import (Compose, VideoRandomCrop, RandomHorizontalFlip, ToTensor)
from Dataloader.mean import get_mean
from torchvision.datasets import UCF101

class CONFIG():
    def __init__(self):
        self.model='dvd-gan'
        self.adv_loss='hinge'
        self.imsize=128
        self.g_num=5
        self.g_chn=32
        self.z_dim=120
        self.ds_chn=32 # SpatialDiscriminator channel
        self.dt_chn=32 # TemporalDiscriminator channel
        self.g_conv_dim=64
        self.d_conv_dim=64
        self.lambda_gp=10.0
        self.lr_schr='const'
        self.version=''
            # Training setting
        self.total_epoch=50
        self.d_iters=1
        self.g_iters=1
        self.batch_size=2
        self.num_workers=1
        self.g_lr=5e-1
        self.d_lr=1e-5
        self.lr_decay=0.9999
        self.beta1=0.0
        self.beta2=0.9

        # using pretrained
        self.pretrained_model = False

        # Misc
        self.train=True
        self.parallel=False
        self.gpus=["0"]
        self.dataset='ucf101'
        self.use_tensorboard=False
        self.n_class=1
        self.k_sample=64
        self.n_frames=24
        self.test_batch_size=8

        # Path
        self.image_path='/home/hlcv_team014/ablation1/VideoCompression_WithoutMaskGenerator/data'
        self.log_path='/home/hlcv_team014/ablation1/VideoCompression_WithoutMaskGenerator/output/logs'
        self.model_save_path='/home/hlcv_team014/ablation1/VideoCompression_WithoutMaskGenerator/output/models'
        self.sample_path='/home/hlcv_team014/ablation1/VideoCompression_WithoutMaskGenerator/output/samples'
        # epoch size
        self.log_epoch=1
        self.sample_epoch=1
        self.model_save_epoch=5000

        # Dataloader
        self.norm_value=255
        self.no_mean_norm=True
        self.std_norm=False
        self.mean_dataset='activitynet'
        self.root_path='/home/hlcv_team014/ablation1/VideoCompression_WithoutMaskGenerator/content/UCF-101'
        self.video_path='videos_jpeg'
        self.annotation_path='/home/hlcv_team014/ablation1/VideoCompression_WithoutMaskGenerator/content/ucfTrainTestlist'
        self.train_crop='none'
        self.sample_size=64

        self.initial_scale=1.0
        self.n_scales=5
        self.scale_step=0.84089641525
        self.density = 0.10

def custom_collate(batch):
    filtered_batch = []
    for video, _, _ in batch:
        filtered_batch.append(video)
    return torch.utils.data.dataloader.default_collate(filtered_batch)

cudnn.benchmark = True

config = CONFIG()

config.dataset = 'ucf101'
config.video_path = "videos_jpeg"
config.adv_loss = 'hinge'
config.parallel = False
config.gpus = ["1"]
config.num_workers = 1
config.use_tensorboard = False 
config.ds_chn = 32 
config.dt_chn = 32 
config.g_chn = 32 
config.n_frames = 8 
config.k_sample = 4 
config.batch_size = 2
config.g_lr = 5e-5 
config.d_lr = 1e-5 
config.n_class = 1 
config.lr_schr = "const"
config.n_class = 2
config.root_path = "/home/hlcv_team014/ablation1/VideoCompression_WithoutMaskGenerator/content/UCF-101"
config.annotation_path = "/home/hlcv_team014/ablation1/VideoCompression_WithoutMaskGenerator/content/ucfTrainTestlist"
config.log_path = "/home/hlcv_team014/ablation1/VideoCompression_WithoutMaskGenerator/output/logs"
config.model_save_path = "/home/hlcv_team014/ablation1/VideoCompression_WithoutMaskGenerator/output/models"
config.sample_path = "/home/hlcv_team014/ablation1/VideoCompression_WithoutMaskGenerator/output/samples"
config.train_crop == 'none'



##### Dataloader #####
config.mean = get_mean(config.norm_value, dataset=config.mean_dataset)

if config.train:
    spatial_transform = Compose([ToTensor(), VideoRandomCrop(128), RandomHorizontalFlip()]) 
    temporal_transform = None
    target_transform = None

    print("="*30,"\nLoading data...")
    training_data = UCF101(config.root_path, config.annotation_path, frames_per_clip=20,
                       step_between_clips=1, train=True, output_format="TCHW", transform=spatial_transform)
    train_loader = torch.utils.data.DataLoader(
            training_data,
            batch_size=config.batch_size,
            collate_fn=custom_collate,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True)


config.total_epoch=50
trainer = Trainer(train_loader, config) 
trainer.train()
