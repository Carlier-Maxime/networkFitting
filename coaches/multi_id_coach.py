import torch
import numpy as np
from tqdm import tqdm
from configs import hyperparameters
from coaches.base_coach import BaseCoach
from PIL import Image

class MultiIDCoach(BaseCoach):

    def __init__(self, device:torch.device, data_loader, network_path, outdir:str='out', save_latent:bool=False, save_video_latent:bool=False, save_video_pti:bool=False, save_img_result:bool=False, seed:int=64, G=None, verbose:bool=True, load_w_pivot:bool=False):
        super().__init__(device, data_loader, network_path, outdir, save_latent, save_video_latent, save_video_pti, save_img_result, seed, G, verbose, load_w_pivot)

    def train(self, first_inv_steps:int=1000, inv_steps:int=100, pti_steps:int=500, max_images:int=-1, paste_color:bool=False, color:torch.Tensor=torch.tensor([-1.,1.,-1.]), epsilon=1.0, save_img_step:bool=False):
        self.G.synthesis.train()
        self.G.mapping.train()

        use_ball_holder = True
        w_pivots = []
        images = []

        if max_images==-1 or max_images>len(self.data_loader.dataset): max_images = len(self.data_loader.dataset)

        self.open_videos()
        for fname, image in tqdm(self.data_loader, desc='calcul latents', unit='image', disable=(not self.verbose), total=max_images):
            if self.image_counter >= max_images: break
            image_name = fname[0]
            w_pivot = self.get_inversions(image_name, image, (inv_steps if len(w_pivots)>0 else first_inv_steps), w_start_pivot=(w_pivots[-1] if len(w_pivots)>0 else None), seed=self.seed, paste_color=paste_color, color=color, epsilon=epsilon, save_img_step=save_img_step)
            if self.save_latent: torch.save(w_pivot, f'{self.outdir}/{image_name}.pt')
            w_pivots.append(w_pivot)
            images.append((image_name, image))
            self.image_counter += 1
        
        seed_images = []
        target_images = []
        step_loss = []
        for i in (pbar := tqdm(range(pti_steps), desc='pivotal tuning', unit='step', postfix='loss: ?', disable=(not self.verbose))):
            self.image_counter = 0
            if self.save_video_pti:
                synth_images = self.forward(self.w_seed)
                synth_images = (synth_images + 1) * (255/2)
                synth_images_np = synth_images.clone().detach().permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                seed_images.append(synth_images_np)
                del synth_images

                synth_images = self.forward(w_pivots[0][0].repeat(1,self.G.num_ws,1))
                synth_images = (synth_images + 1) * (255/2)
                synth_images_np = synth_images.clone().detach().permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                target_images.append(synth_images_np)
                del synth_images

            training_step = 1
            for data, w_pivot in tqdm(zip(images, w_pivots), total=len(images), desc='PTI', unit='image', disable=(not self.verbose) or len(images)<10):
                image_name, image = data
                if self.image_counter >= max_images: break

                real_images_batch = image.to(self.device)
                synth_images = self.forward(w_pivot[0].repeat(1,self.G.num_ws,1))
                loss, l2_loss_val, loss_lpips = self.calc_loss(synth_images, real_images_batch, image_name, self.G, use_ball_holder, w_pivot)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                use_ball_holder = training_step % hyperparameters.locality_regularization_interval == 0

                training_step += 1
                self.image_counter += 1
                loss_val = float(loss)
                step_loss.append(loss_val)
            pbar.set_postfix_str(f'loss: {np.mean(step_loss):<5.2f}')

        if self.save_video_pti:
            self.save_imgs_to_video(seed_images, "fitting_seed.mp4", "Saving network fitting video (seed view)..", fps=60)
            self.save_imgs_to_video(target_images, "fitting_target.mp4", "Saving network fitting video (target view)..", fps=60)
            del seed_images
            del target_images

        self.close_videos()

        if self.save_img_result:
            self.image_counter = 0
            for data, w_pivot in tqdm(zip(images, w_pivots), total=len(images), desc='save image result', unit='image', disable=(not self.verbose)):
                image_name, image = data
                if self.image_counter >= max_images: break
                real_images_batch = image.to(self.device)
                synth_images = self.forward(w_pivot[0].repeat(1,self.G.num_ws,1))
                synth_images = (synth_images + 1) * (255/2)
                synth_images_np = synth_images.clone().detach().permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                Image.fromarray(synth_images_np, 'RGB').save(f'{self.outdir}/project_{image_name}.png')
                self.image_counter += 1

        torch.save(self.G, f'{self.outdir}/network.pt')
