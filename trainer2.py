import time
import torch
import datetime

import torch.nn as nn
from torchvision.utils import save_image, make_grid
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau, StepLR, MultiStepLR

from Module.Generator import Generator, Mask_Generator_Net, Generator_Net
from Module.Discriminators import SpatialDiscriminator, TemporalDiscriminator, DiscriminatorLSTM, Discriminator_space
from utils import *


class Trainer(object):
    def __init__(self, data_loader, config):

        # Data loader
        self.data_loader = data_loader

        # exact model and loss
        self.model = config.model
        self.adv_loss = config.adv_loss

        # Model hyper-parameters
        self.imsize = config.imsize
        self.g_num = config.g_num
        self.z_dim = config.z_dim
        self.g_chn = config.g_chn
        self.ds_chn = config.ds_chn
        self.dt_chn = config.dt_chn
        self.n_frames = config.n_frames
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.lr_schr = config.lr_schr

        self.lambda_gp = config.lambda_gp
        self.total_epoch = config.total_epoch
        self.d_iters = config.d_iters
        self.g_iters = config.g_iters
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.lr_decay = config.lr_decay
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.pretrained_model = config.pretrained_model

        self.n_class = config.n_class
        self.k_sample = config.k_sample
        self.dataset = config.dataset
        self.use_tensorboard = config.use_tensorboard
        self.test_batch_size = config.test_batch_size
        self.density = config.density

        # path
        self.image_path = config.image_path
        self.log_path = config.log_path
        self.model_save_path = config.model_save_path
        self.sample_path = config.sample_path

        # epoch size
        self.log_epoch = config.log_epoch
        self.sample_epoch = config.sample_epoch
        self.model_save_epoch = config.model_save_epoch
        self.version = config.version

        # Path
        self.log_path = os.path.join(config.log_path, self.version)
        self.sample_path = os.path.join(config.sample_path, self.version)
        self.model_save_path = os.path.join(config.model_save_path, self.version)

        self.device, self.parallel, self.gpus = set_device(config)

        self.build_model()

        if self.use_tensorboard:
            self.build_tensorboard()

        # Start with trained model
        if self.pretrained_model:
            print('load_pretrained_model...')
            self.load_pretrained_model()

    def label_sample(self):
        label = torch.randint(low=0, high=self.n_class, size=(self.batch_size, ))
        # label = torch.LongTensor(self.batch_size, 1).random_()%self.n_class
        # one_hot= torch.zeros(self.batch_size, self.n_class).scatter_(1, label, 1)
        return label.to(self.device)  # , one_hot.to(self.device)

    def wgan_loss(self, real_img, fake_img, tag):

        # Compute gradient penalty
        alpha = torch.rand(real_img.size(0), 1, 1, 1).cuda().expand_as(real_img)
        interpolated = torch.tensor(alpha * real_img.data + (1 - alpha) * fake_img.data, requires_grad=True)
        if tag == 'S':
            #out = self.D_s(interpolated)
        else:
            out = self.D_t(interpolated)
        grad = torch.autograd.grad(outputs=out,
                                   inputs=interpolated,
                                   grad_outputs=torch.ones(out.size()).cuda(),
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        grad = grad.view(grad.size(0), -1)
        grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
        d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)

        # Backward + Optimize
        loss = self.lambda_gp * d_loss_gp
        return loss

    def calc_loss(self, x, real_flag):
        if real_flag is True:
            x = -x
        if self.adv_loss == 'wgan-gp':
            loss = torch.mean(x)
        elif self.adv_loss == 'hinge':
            loss = torch.nn.ReLU()(1.0 + x).mean()
        return loss
    
    def densityLoss(self, mask):
        new_mask = mask.view(-1)  ##Binary mask output from the Mask Generator
        pixel_selected = ((torch.sum(new_mask)/ new_mask.shape[0]) - self.density)**2
        return pixel_selected

    def gen_real_video(self, data_iter):

        try:
            real_videos, real_labels = next(data_iter)
        except:
            data_iter = iter(self.data_loader)
            real_videos, real_labels = next(data_iter)
            self.epoch += 1

        return real_videos.to(self.device), real_labels.to(self.device)

    def select_opt_schr(self):

#         self.g_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.G.parameters()), self.g_lr,
#                                             (self.beta1, self.beta2))
        self.g_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, list(self.G.parameters())+list(self.mask_G.parameters())), self.g_lr,
                                            (self.beta1, self.beta2))
        # self.ds_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D_s.parameters()), self.d_lr,
        #                                      (self.beta1, self.beta2))
        self.dt_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D_t.parameters()), self.d_lr,
                                             (self.beta1, self.beta2))
        if self.lr_schr == 'const':
            self.g_lr_scher = StepLR(self.g_optimizer, step_size=10000, gamma=1)
            #self.ds_lr_scher = StepLR(self.ds_optimizer, step_size=10000, gamma=1)
            self.dt_lr_scher = StepLR(self.dt_optimizer, step_size=10000, gamma=1)
        elif self.lr_schr == 'step':
            self.g_lr_scher = StepLR(self.g_optimizer, step_size=500, gamma=0.98)
            #self.ds_lr_scher = StepLR(self.ds_optimizer, step_size=500, gamma=0.98)
            self.dt_lr_scher = StepLR(self.dt_optimizer, step_size=500, gamma=0.98)
        elif self.lr_schr == 'exp':
            self.g_lr_scher = ExponentialLR(self.g_optimizer, gamma=0.9999)
            #self.ds_lr_scher = ExponentialLR(self.ds_optimizer, gamma=0.9999)
            self.dt_lr_scher = ExponentialLR(self.dt_optimizer, gamma=0.9999)
        elif self.lr_schr == 'multi':
            self.g_lr_scher = MultiStepLR(self.g_optimizer, [10000, 30000], gamma=0.3)
            #self.ds_lr_scher = MultiStepLR(self.ds_optimizer, [10000, 30000], gamma=0.3)
            self.dt_lr_scher = MultiStepLR(self.dt_optimizer, [10000, 30000], gamma=0.3)
        else:
            self.g_lr_scher = ReduceLROnPlateau(self.g_optimizer, mode='min',
                                                factor=self.lr_decay, patience=100,
                                                threshold=0.0001, threshold_mode='rel',
                                                cooldown=0, min_lr=1e-10, eps=1e-08,
                                                verbose=True
                            )
            #self.ds_lr_scher = ReduceLROnPlateau(self.ds_optimizer, mode='min',
                                                 factor=self.lr_decay, patience=100,
                                                 threshold=0.0001, threshold_mode='rel',
                                                 cooldown=0, min_lr=1e-10, eps=1e-08,
                                                 verbose=True
                             )
            self.dt_lr_scher = ReduceLROnPlateau(self.dt_optimizer, mode='min',
                                                 factor=self.lr_decay, patience=100,
                                                 threshold=0.0001, threshold_mode='rel',
                                                 cooldown=0, min_lr=1e-10, eps=1e-08,
                                                 verbose=True
                             )

    def epoch2step(self):

        self.epoch = 0
        step_per_epoch = len(self.data_loader)
        print("steps per epoch:", step_per_epoch)

        self.total_step = self.total_epoch * step_per_epoch
        self.log_step = self.log_epoch * step_per_epoch
        self.sample_step = self.sample_epoch * step_per_epoch
        self.model_save_step = self.model_save_epoch

    def train(self):

        # Data iterator
        print("Inside the trainer!")
        data_iter = iter(self.data_loader)
        print("Iterator created!")
        self.epoch2step()

        #fixed_z = torch.randn(self.test_batch_size * self.n_class, self.z_dim).to(self.device)
        # fixed_label = torch.randint(low=0, high=self.n_class, size=(self.test_batch_size, )).to(self.device)
        #fixed_label = torch.tensor([i for i in range(self.n_class) for j in range(self.test_batch_size)])

        # Start with trained model
        if self.pretrained_model:
            start = self.pretrained_model + 1
        else:
            start = 1

        # Start time
        print("=" * 30, "\nStart training...")
        start_time = time.time()

        #self.D_s.train()
        self.D_t.train()
        self.G.train()
        self.mask_G.train()

        for step in range(start, self.total_step):

            # real_videos, real_labels = self.gen_real_video(data_iter)
            if step%200==0:
                print(step)
            try:
                real_videos = next(data_iter)
                #print(real_videos.shape)
            except:
                data_iter = iter(self.data_loader)
                real_videos = next(data_iter)
                self.epoch += 1

            real_videos = real_videos.to(self.device)

            # B x C x T x H x W --> B x T x C x H x W
            # real_videos = real_videos.permute(0, 2, 1, 3, 4).contiguous()

            # ================ update D d_iters times ================ #
            for i in range(self.d_iters):
                
                # ============== Genrate Masks =================== #
                #print("Started")
                
                B,T,C,H,W = real_videos.shape
                seed1 = torch.rand((B,T,C,H,W))
                video_mask = self.mask_G(torch.cat((real_videos, seed1), 2))  
                torch.cuda.empty_cache()
                
                #print("Mask generated")
                # ============== Create Masked Video ============== #
                B,T,C,H,W = video_mask.shape
                masked_video = video_mask.repeat(1,1,3,1,1) * real_videos
                torch.cuda.empty_cache()
                
                # ============== Frame Generator ================= #
                
                seed2 = torch.rand((B,T,C,H,W))
                fake_video = self.G(torch.cat((masked_video, video_mask, seed2), 2))              
                torch.cuda.empty_cache()
                #print("Inpainted Video Generated")
                
                fake_input = torch.cat((fake_video, video_mask), 2)
                real_input = torch.cat((real_videos, video_mask), 2)
                if(step%10==0):
                    # # ============== Discriminator_spatial =========== #                
                    
                    # d_s_fake = self.D_s(fake_input.detach())
                    # torch.cuda.empty_cache()
                    # d_s_real = self.D_s(real_input.detach())
                    # torch.cuda.empty_cache()
                    # ds_loss_real = self.calc_loss(d_s_real, True)
                    # ds_loss_fake = self.calc_loss(d_s_fake, False)
                    
                    
                    # # ============== Optimize D_s ============= #
                    
                    # ds_loss = ds_loss_real + ds_loss_fake
                    # if step%200 ==0:
                    #     print("DS_Loss:"+str(ds_loss))
                    # self.reset_grad()
                    # ds_loss.backward()
                    # self.G.zero_grad()
                    # self.mask_G.zero_grad()
                    # self.ds_optimizer.step()
                    # self.ds_lr_scher.step()
                    # #print("Spatial Discrimination done")
                    
                    # ============== Discriminator Temporal ========== #
                    
                    d_t_fake = self.D_t(fake_input.detach())
                    d_t_real = self.D_t(real_input.detach())
                    dt_loss_real = self.calc_loss(d_t_real, True)
                    dt_loss_fake = self.calc_loss(d_t_fake, False)
                    
                    
                    # ============== Optimize D_t ================== #
                    
                    dt_loss = dt_loss_real + dt_loss_fake
                    if step%200 ==0:
                        print("DT_Loss:"+str(dt_loss))
                    self.reset_grad()
                    dt_loss.backward()
                    self.G.zero_grad()
                    self.mask_G.zero_grad()
                    self.dt_optimizer.step()
                    self.dt_lr_scher.step()
                    #print("Temporal Discrimination done")
                
                # ============== Generator and Mask Generator ========= #
                
                # d_s_fake = self.D_s(fake_input)
                d_t_fake = self.D_t(fake_input)
                # ds_loss_fake = self.calc_loss(d_s_fake, True)
                dt_loss_fake = self.calc_loss(d_t_fake, True)
                mse_loss = nn.MSELoss()(fake_video, real_videos)
                # d_loss = self.densityLoss(video_mask)
                new_mask = video_mask.view(-1)  ##Binary mask output from the Mask Generator
                selected_density = ((torch.sum(new_mask)/ new_mask.shape[0]))
                if(selected_density<self.density):
                    d_loss = 10*((selected_density - self.density)/self.density)**2
                else:
                    d_loss = (selected_density - self.density)**2
                
                if step%200 ==0:
                    print("Selected Density:"+str(selected_density))
                    print("DS_Loss:"+str(ds_loss_fake)+" DT_Loss:"+str(dt_loss_fake)+" MSE_Loss:"+str(mse_loss)+" Density_Loss:"+str(d_loss))
                #g_loss = ds_loss_fake + dt_loss_fake + mse_loss + d_loss
                g_loss = dt_loss_fake + 5*mse_loss + d_loss
                
                # ============== Optimize G and mask_G ================ #
                
                self.reset_grad()
                g_loss.backward()
                #self.D_s.zero_grad()
                self.D_t.zero_grad()
                self.g_optimizer.step()
                self.g_lr_scher.step()
                
                #print("Generator Done")
                
                
                
                
                
                
                
                
                

#                 # ============= Generate real video ============== #
#                 real_videos_sample = sample_k_frames(real_videos, self.n_frames, self.k_sample)

#                 # ============= Generate fake video ============== #
#                 # apply Gumbel Softmax
#                 z = torch.randn(self.batch_size, self.z_dim).to(self.device)
#                 z_class = self.label_sample()
#                 fake_videos = self.G(z, z_class)

#                 # ================== Train D_s ================== #
#                 fake_videos_sample = sample_k_frames(fake_videos, self.n_frames, self.k_sample)
#                 ds_out_real = self.D_s(real_videos_sample, real_labels)
#                 ds_out_fake = self.D_s(fake_videos_sample.detach(), z_class)
#                 ds_loss_real = self.calc_loss(ds_out_real, True)
#                 ds_loss_fake = self.calc_loss(ds_out_fake, False)

#                 # Backward + Optimize
#                 ds_loss = ds_loss_real + ds_loss_fake
#                 self.reset_grad()
#                 ds_loss.backward()
#                 self.ds_optimizer.step()
#                 self.ds_lr_scher.step()

#                 # ================== Train D_t ================== #
#                 real_videos_downsample = vid_downsample(real_videos)
#                 fake_videos_downsample = vid_downsample(fake_videos)

#                 dt_out_real = self.D_t(real_videos_downsample, real_labels)
#                 dt_out_fake = self.D_t(fake_videos_downsample.detach(), z_class)
#                 dt_loss_real = self.calc_loss(dt_out_real, True)
#                 dt_loss_fake = self.calc_loss(dt_out_fake, False)

#                 # Backward + Optimize
#                 dt_loss = dt_loss_real + dt_loss_fake
#                 self.reset_grad()
#                 dt_loss.backward()
#                 self.dt_optimizer.step()
#                 self.dt_lr_scher.step()

                # ================== Use wgan_gp ================== #
                # if self.adv_loss == "wgan_gp":
                #     dt_wgan_loss = self.wgan_loss(real_labels, fake_videos, 'T')
                #     ds_wgan_loss = self.wgan_loss(real_labels, fake_videos, 'S')
                #     self.reset_grad()
                #     dt_wgan_loss.backward()
                #     ds_wgan_loss.backward()
                #     self.dt_optimizer.step()
                #     self.ds_optimizer.step()

            # ==================== update G g_iters time ==================== #

            # for i in range(self.g_iters):

                # ============= Generate fake video ============== #
                # apply Gumbel Softmax
                # if i > 1:
                #     z = torch.randn(self.batch_size, self.z_dim).to(self.device)
                #     z_class = self.label_sample()
                #     fake_videos = self.G(z, z_class)
                #     fake_videos_sample = sample_k_frames(fake_videos, self.n_frames, self.k_sample)
                #     fake_videos_downsample = vid_downsample(fake_videos)

            # =========== Train G and Gumbel noise =========== #
            # Compute loss with fake images
#             g_s_out_fake = self.D_s(fake_videos_sample, z_class)  # Spatial Discrimminator loss
#             g_t_out_fake = self.D_t(fake_videos_downsample, z_class)  # Temporal Discriminator loss
#             g_s_loss = self.calc_loss(g_s_out_fake, True)
#             g_t_loss = self.calc_loss(g_t_out_fake, True)
#             g_loss = g_s_loss + g_t_loss
#             # g_loss = self.calc_loss(g_s_out_fake, True) + self.calc_loss(g_t_out_fake, True)

#             # Backward + Optimize
#             self.reset_grad()
#             g_loss.backward()
#             self.g_optimizer.step()
#             self.g_lr_scher.step()

            # ==================== print & save part ==================== #
            # Print out log info
            if step % self.log_step == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                start_time = time.time()
                # log_str = "Epoch: [%d/%d], Step: [%d/%d], time: %s, ds_loss: %.4f, dt_loss: %.4f, g_loss: %.4f, lr: %.2e" % \
                #     (self.epoch, self.total_epoch, step, self.total_step, elapsed, ds_loss, dt_loss, g_loss, self.g_lr_scher.get_lr()[0])
                log_str = "Epoch: [%d/%d], Step: [%d/%d], time: %s, ds_loss: %.4f, dt_loss: %.4f, g_loss: %.4f, lr: %.2e" % \
                    (self.epoch, self.total_epoch, step, self.total_step, elapsed, dt_loss, g_loss, self.g_lr_scher.get_lr()[0])

                if self.use_tensorboard is True:
                    # write_log(self.writer, log_str, step, ds_loss_real, ds_loss_fake, ds_loss, dt_loss_real, dt_loss_fake, dt_loss, g_loss)
                    write_log(self.writer, log_str, step, dt_loss_real, dt_loss_fake, dt_loss, g_loss)
                print(log_str)

            # Sample images
            # if step % self.sample_step == 0:
            #     self.G.eval()
            #     fake_videos = self.G(fixed_z, fixed_label)

            #     for i in range(self.n_class):
            #         for j in range(self.test_batch_size):
            #             if self.use_tensorboard is True:
            #                 self.writer.add_image("Class_%d_No.%d/Step_%d" % (i, j, step), make_grid(denorm(fake_videos[i * self.test_batch_size + j].data)), step)
            #             else:
            #                 save_image(denorm(fake_videos[i * self.test_batch_size + j].data), os.path.join(self.sample_path, "Class_%d_No.%d_Step_%d" % (i, j, step)))
            #     # print('Saved sample images {}_fake.png'.format(step))
            #     self.G.train()

            # Save model
            if step % self.model_save_step == 0:
                torch.save(self.G.state_dict(),
                           os.path.join(self.model_save_path, '{}_G.pth'.format(step)))
                torch.save(self.mask_G.state_dict(),
                           os.path.join(self.model_save_path, '{}_mask_G.pth'.format(step)))
                # torch.save(self.D_s.state_dict(),
                #           os.path.join(self.model_save_path, '{}_Ds.pth'.format(step)))
                torch.save(self.D_t.state_dict(),
                           os.path.join(self.model_save_path, '{}_Dt.pth'.format(step)))

    def build_model(self):

        print("=" * 30, '\nBuild_model...')

#         self.G = Generator(self.z_dim, n_class=self.n_class, ch=self.g_chn, n_frames=self.n_frames).cuda()
#         self.D_s = SpatialDiscriminator(chn=self.ds_chn, n_class=self.n_class).cuda()
#         self.D_t = TemporalDiscriminator(chn=self.dt_chn, n_class=self.n_class).cuda()
        
        
        self.mask_G = Mask_Generator_Net(6).to(self.device)
        self.G = Generator_Net(7).to(self.device)
        self.D_t = DiscriminatorLSTM(4).to(self.device)
        #self.D_s = Discriminator_space(4).to(self.device)

        if self.parallel:
            print('Use parallel...')
            print('gpus:', os.environ["CUDA_VISIBLE_DEVICES"])

#             self.G = nn.DataParallel(self.G, device_ids=self.gpus)
#             self.D_s = nn.DataParallel(self.D_s, device_ids=self.gpus)
#             self.D_t = nn.DataParallel(self.D_t, device_ids=self.gpus)
            self.mask_G = nn.DataParallel(self.mask_G, device_ids=self.gpus)
            self.G = nn.DataParallel(self.G, device_ids=self.gpus)
            # self.D_s = nn.DataParallel(self.D_s, device_ids=self.gpus)
            self.D_t = nn.DataParallel(self.D_t, device_ids=self.gpus)

        # self.G.apply(weights_init)
        # self.D.apply(weights_init)

        self.select_opt_schr()

        self.c_loss = torch.nn.CrossEntropyLoss()
        print("Model building done!")

    def build_tensorboard(self):
        from tensorboardX import SummaryWriter
        # from logger import Logger
        # self.logger = Logger(self.log_path)

        self.writer = SummaryWriter(log_dir=self.log_path)

    def load_pretrained_model(self):
        self.G.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_G.pth'.format(self.pretrained_model))))
        self.mask_G.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_mask_G.pth'.format(self.pretrained_model))))
        # self.D_s.load_state_dict(torch.load(os.path.join(
        #     self.model_save_path, '{}_Ds.pth'.format(self.pretrained_model))))
        self.D_t.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_Dt.pth'.format(self.pretrained_model))))
        print('loaded trained models (step: {})..!'.format(self.pretrained_model))

    def reset_grad(self):
        #self.ds_optimizer.zero_grad()
        self.dt_optimizer.zero_grad()
        self.g_optimizer.zero_grad()

    def save_sample(self, data_iter):
        real_images, _ = next(data_iter)
        save_image(denorm(real_images), os.path.join(self.sample_path, 'real.png'))
