import imageio
import torch
from tqdm import tqdm
from configs import hyperparameters
from coaches.base_coach import BaseCoach
from PIL import Image

class SingleIDCoach(BaseCoach):

    def __init__(self, device:torch.device, data_loader, network_path, outdir:str='out', save_latent:bool=False, save_video_latent:bool=False, save_video_pti:bool=False, save_img_result:bool=False, seed:int=64, G=None, verbose:bool=True):
        super().__init__(device, data_loader, network_path, outdir, save_latent, save_video_latent, save_video_pti, save_img_result, seed, G, verbose)

    def train(self, first_inv_steps:int=1000, inv_steps:int=100, pti_steps:int=500, max_images:int=-1, paste_color:bool=False, color:torch.Tensor=torch.tensor([-1.,1.,-1.]), epsilon=1.0, save_img_step:bool=False):
        use_ball_holder = True
        w_pivot = None
        if max_images==-1 or max_images>len(self.data_loader.dataset): max_images = len(self.data_loader.dataset)
        pbar1 = tqdm(total=max_images, desc='fitting', unit='image', disable=(not self.verbose))
        pbar2 = tqdm(total=first_inv_steps, desc='optimization Latent', unit='step', disable=(not self.verbose))
        pbar3 = tqdm(total=pti_steps, desc='pivotal tuning', unit='step', disable=(not self.verbose))
        if self.save_video_latent:
            videoOptiLatent = imageio.get_writer(f'{self.outdir}/optiLatent.mp4', mode='I', fps=60, codec='libx264', bitrate='16M')
            videoResultLatent = imageio.get_writer(f'{self.outdir}/resultLatent.mp4', mode='I', fps=10, codec='libx264', bitrate='16M')
        if self.save_video_pti:
            videoFittingSeed = imageio.get_writer(f'{self.outdir}/fitting_seed.mp4', mode='I', fps=60, codec='libx264', bitrate='16M')
            videoFittingTarget = imageio.get_writer(f'{self.outdir}/fitting_target.mp4', mode='I', fps=60, codec='libx264', bitrate='16M')
        for fname, image in self.data_loader:
            w_imgs=[]
            wrimgs=[]
            seed_images=[]
            target_images=[]
            image_name = fname[0]
            if self.image_counter >= max_images: break
            if hyperparameters.use_last_w_pivots:
                w_pivot = self.load_inversions(image_name)
            elif not hyperparameters.use_last_w_pivots or w_pivot is None:
                imgs, w_pivot = self.calc_inversions(image, (inv_steps if w_pivot!=None else first_inv_steps), w_start_pivot=w_pivot, seed=self.seed, paste_color=paste_color, color=color, epsilon=epsilon, save_img_step=save_img_step, pbar=pbar2)
                if self.save_video_latent:
                    w_imgs += imgs
                    wrimgs.append(imgs[-1])

            w_pivot = w_pivot.to(self.device)
            if self.save_latent: torch.save(w_pivot, f'{self.outdir}/{image_name}.pt')
            real_images_batch = image.to(self.device)

            generated_images = None
            for i in range(pti_steps):
                generated_images = self.forward(w_pivot[0].repeat(1,self.G.num_ws,1))
                if self.save_video_pti:
                    synth_images = self.forward(self.w_seed)
                    synth_images = (synth_images + 1) * (255/2)
                    synth_images_np = synth_images.clone().detach().permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                    seed_images.append(synth_images_np)
                    del synth_images

                    synth_images = (generated_images + 1) * (255/2)
                    synth_images_np = synth_images.clone().detach().permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                    target_images.append(synth_images_np)
                    del synth_images
                loss, l2_loss_val, loss_lpips = self.calc_loss(generated_images, real_images_batch, image_name, self.G, use_ball_holder, w_pivot)

                self.optimizer.zero_grad()
                if loss_lpips <= hyperparameters.LPIPS_value_threshold: break

                loss.backward()
                self.optimizer.step()
                use_ball_holder = (i+1) % hyperparameters.locality_regularization_interval == 0

                pbar3.update()
                pbar3.set_postfix_str(f'loss: {float(loss):<5.2f}')
            self.image_counter += 1
            if self.save_img_result:
                if generated_images is None: generated_images_np = wrimgs[-1]
                else:
                    generated_images = (generated_images + 1) * (255/2)
                    generated_images_np = generated_images.clone().detach().permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                Image.fromarray(generated_images_np, 'RGBA' if generated_images_np.shape[2]==4 else 'RGB').save(f'{self.outdir}/project_{image_name}.png')
            del generated_images
            if pbar1.n==0:
                pbar2.total = inv_steps
                pbar2.n = inv_steps
            if pbar1.n!=max_images-1:
                pbar3.reset()
                pbar2.reset()
            pbar1.update()
            if self.save_video_latent:
                self.video_append(videoOptiLatent, w_imgs)
                self.video_append(videoResultLatent, wrimgs)

            if self.save_video_pti:
                self.video_append(videoFittingSeed, seed_images)
                self.video_append(videoFittingTarget, target_images)
        pbar1.close()
        pbar2.close()
        pbar3.close()
        if self.save_video_latent:
            videoOptiLatent.close()
            videoResultLatent.close()

        if self.save_video_pti:
            videoFittingSeed.close()
            videoResultLatent.close()

        if pti_steps>0: torch.save(self.G, f'{self.outdir}/network.pt')
