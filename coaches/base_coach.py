import abc
import os
import os.path

import imageio
import numpy as np
import torch
from lpips import LPIPS

from configs import hyperparameters
from criteria import l2_loss
from criteria.localitly_regulizer import Space_Regulizer
from stylegan_xl.run_inversion import project
from stylegan_xl.torch_utils import gen_utils
from utils.models_utils import toogle_grad, load_network


class BaseCoach:
    def __init__(self, device: torch.device, data_loader, network_path, outdir, save_latent: bool = False, save_video_latent: bool = False, save_video_pti: bool = False, save_img_result: bool = False, seed: int = 64, G=None, verbose: bool = True, load_w_pivot: bool = False):
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
        self.load_w_pivot = load_w_pivot
        os.makedirs(self.outdir, exist_ok=True)

    def restart_training(self, G=None):

        # Initialize networks
        self.G = load_network(self.network_path, self.device) if G == None else G
        toogle_grad(self.G, True)

        self.original_G = load_network(self.network_path, self.device)

        self.space_regulizer = Space_Regulizer(self.original_G, self.lpips_loss)
        self.optimizer = self.configure_optimizers()

    def get_inversions(self, image_name, image, num_steps: int = 1000, w_start_pivot=None, seed: int = 64, paste_color: bool = False, color: torch.Tensor = torch.tensor([-1., 1., -1.]), epsilon=1.0, save_img_step: bool = False, pbar=None):
        if image_name in self.w_pivots: return self.w_pivots[image_name]
        w = self.load_inversions(image_name) if self.load_w_pivot else None
        if w is None:
            imgs, w = project(self.G, image, device=torch.device(self.device), w_avg_samples=600, num_steps=num_steps, w_start_pivot=w_start_pivot, seed=seed, verbose=self.verbose, paste_color=paste_color, color=color, epsilon=epsilon, save_img_step=save_img_step, pbar=pbar)
            if self.save_video_latent:
                self.video_append(self.videoOptiLatent, imgs)
                self.video_append(self.videoResultLatent, [imgs[-1]])
        self.w_pivots[image_name] = w
        return w

    def load_inversions(self, image_name):
        w_path = f'{self.outdir}/latent_{image_name}.pt'
        if not os.path.isfile(w_path): return None
        return torch.load(w_path).to(self.device)

    @abc.abstractmethod
    def train(self):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.G.parameters(), lr=hyperparameters.pti_learning_rate)

        return optimizer

    def calc_loss(self, generated_images, real_images, log_name, new_G, use_ball_holder, w_batch):
        loss = 0.0

        if hyperparameters.pt_l2_lambda > 0:
            l2_loss_val = l2_loss.l2_loss(real_images, generated_images)
            loss += l2_loss_val * hyperparameters.pt_l2_lambda
        if hyperparameters.pt_lpips_lambda > 0:
            loss_lpips = self.lpips_loss(generated_images[:, :3], real_images[:, :3])
            if real_images.shape[1] == 4: loss_lpips += self.lpips_loss(generated_images[:, 3:4].repeat(1, 3, 1, 1), real_images[:, 3:4].repeat(1, 3, 1, 1))
            loss_lpips = torch.squeeze(loss_lpips)
            loss += loss_lpips * hyperparameters.pt_lpips_lambda

        if use_ball_holder and hyperparameters.use_locality_regularization:
            ball_holder_loss_val = self.space_regulizer.space_regulizer_loss(new_G, w_batch)
            loss += ball_holder_loss_val

        return loss, l2_loss_val, loss_lpips

    def forward(self, w):
        generated_images = self.G.synthesis(w, noise_mode='const', force_fp32=True)
        return generated_images

    def save_imgs_to_video(self, imgs, outfile: str, msg: str = "", fps: int = 60):
        if self.verbose and msg != "": print(msg)
        video = imageio.get_writer(f'{self.outdir}/{outfile}', mode='I', fps=fps, codec='libx264', bitrate='16M')
        self.video_append(video, imgs)
        video.close()

    def video_append(self, video, imgs):
        for synth_image in imgs: video.append_data(np.array(synth_image))

    def open_videos(self):
        if self.save_video_latent:
            self.videoOptiLatent = imageio.get_writer(f'{self.outdir}/optiLatent.mp4', mode='I', fps=60, codec='libx264', bitrate='16M')
            self.videoResultLatent = imageio.get_writer(f'{self.outdir}/resultLatent.mp4', mode='I', fps=10, codec='libx264', bitrate='16M')
        if self.save_video_pti:
            self.videoFittingSeed = imageio.get_writer(f'{self.outdir}/fitting_seed.mp4', mode='I', fps=60, codec='libx264', bitrate='16M')
            self.videoFittingTarget = imageio.get_writer(f'{self.outdir}/fitting_target.mp4', mode='I', fps=60, codec='libx264', bitrate='16M')

    def close_videos(self):
        if self.save_video_latent:
            self.videoOptiLatent.close()
            self.videoResultLatent.close()
        if self.save_video_pti:
            self.videoFittingSeed.close()
            self.videoResultLatent.close()
