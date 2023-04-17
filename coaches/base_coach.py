import abc
import os
import pickle
from argparse import Namespace
import os.path
from criteria.localitly_regulizer import Space_Regulizer
import torch
from torchvision import transforms
from lpips import LPIPS
from configs import global_config, paths_config, hyperparameters
from criteria import l2_loss
from utils.models_utils import toogle_grad, load_network
import sys
sys.path.insert(1, 'stylegan-xl')
from run_inversion import project
from torch_utils import gen_utils
import imageio
import numpy as np

class BaseCoach:
    def __init__(self, device:torch.device, data_loader, network_path, outdir, save_latent:bool=False, save_video_latent:bool=False, save_video_pti:bool=False, save_img_result:bool=False, seed:int=64, G=None, verbose:bool=True):
        self.device = device
        self.data_loader = data_loader
        self.network_path = network_path
        self.outdir = outdir
        self.save_latent = save_latent
        self.save_video_latent = save_video_latent
        self.save_video_pti = save_video_pti
        self.save_img_result = save_img_result
        self.w_pivots = {}
        self.image_counter = 0
        self.lpips_loss = LPIPS(net=hyperparameters.lpips_type, verbose=verbose).to(device).eval()
        self.restart_training(G=G)
        self.seed = seed
        self.w_seed = gen_utils.get_w_from_seed(self.G, 1, device, seed=seed)
        self.verbose = verbose
        os.makedirs(self.outdir, exist_ok=True)

    def restart_training(self, G=None):

        # Initialize networks
        self.G = load_network(self.network_path, self.device) if G==None else G
        toogle_grad(self.G, True)

        self.original_G = load_network(self.network_path, self.device)

        self.space_regulizer = Space_Regulizer(self.original_G, self.lpips_loss)
        self.optimizer = self.configure_optimizers()

    def load_inversions(self, image_name):
        if image_name in self.w_pivots:
            return self.w_pivots[image_name]

        w_potential_path = f'{self.outdir}/latent_{image_name}.pt'
        if not os.path.isfile(w_potential_path):
            return None
        w = torch.load(w_potential_path).to(self.device)
        self.w_pivots[image_name] = w
        return w

    def calc_inversions(self, image, num_steps, w_start_pivot=None, seed:int=64, paste_color:bool=False, color:torch.Tensor=torch.tensor([-1.,1.,-1.]), epsilon=1.0, save_img_step:bool=False):
        return project(self.G, image, device=torch.device(self.device), w_avg_samples=600, num_steps=num_steps, w_start_pivot=w_start_pivot, seed=seed, verbose=self.verbose, paste_color=paste_color, color=color, epsilon=epsilon, save_img_step=save_img_step)

    @abc.abstractmethod
    def train(self):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.G.parameters(), lr=hyperparameters.pti_learning_rate)

        return optimizer

    def calc_loss(self, generated_images, real_images, log_name, new_G, use_ball_holder, w_batch):
        loss = 0.0

        if hyperparameters.pt_l2_lambda > 0:
            l2_loss_val = l2_loss.l2_loss(generated_images, real_images)
            loss += l2_loss_val * hyperparameters.pt_l2_lambda
        if hyperparameters.pt_lpips_lambda > 0:
            loss_lpips = self.lpips_loss(generated_images, real_images)
            loss_lpips = torch.squeeze(loss_lpips)
            loss += loss_lpips * hyperparameters.pt_lpips_lambda

        if use_ball_holder and hyperparameters.use_locality_regularization:
            ball_holder_loss_val = self.space_regulizer.space_regulizer_loss(new_G, w_batch)
            loss += ball_holder_loss_val

        return loss, l2_loss_val, loss_lpips

    def forward(self, w):
        generated_images = self.G.synthesis(w, noise_mode='const', force_fp32=True)
        return generated_images
    
    def save_imgs_to_video(self, imgs, outfile:str, msg:str="", fps:int=60):
        if self.verbose: print(msg)
        video = imageio.get_writer(f'{self.outdir}/{outfile}', mode='I', fps=fps, codec='libx264', bitrate='16M')
        for synth_image in imgs: video.append_data(np.array(synth_image))
        video.close()
