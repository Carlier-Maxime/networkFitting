import os
import torch
from tqdm import tqdm
from configs import paths_config, hyperparameters, global_config
from training.coaches.base_coach import BaseCoach

class SingleIDCoach(BaseCoach):

    def __init__(self, data_loader, network_path, outdir:str='out', save_latent:bool=False, save_video_latent:bool=False, save_video_pti:bool=False):
        super().__init__(data_loader, network_path, outdir, save_latent, save_video_latent, save_video_pti)

    def train(self, first_inv_steps:int=1000, inv_steps:int=100, pti_steps:int=500):
        use_ball_holder = True
        for fname, image in tqdm(self.data_loader):
            image_name = fname[0]
            self.restart_training()
            if self.image_counter >= hyperparameters.max_images_to_invert:
                break
            w_pivot = None
            if hyperparameters.use_last_w_pivots:
                w_pivot = self.load_inversions(image_name)
            elif not hyperparameters.use_last_w_pivots or w_pivot is None:
                w_pivot = self.calc_inversions(image, (inv_steps if len(w_pivots)>0 else first_inv_steps))

            # w_pivot = w_pivot.detach().clone().to(global_config.device)
            w_pivot = w_pivot.to(global_config.device)

            torch.save(w_pivot, f'{embedding_dir}/0.pt')
            real_images_batch = image.to(global_config.device)

            for i in tqdm(range(pti_steps)):

                generated_images = self.forward(w_pivot)
                loss, l2_loss_val, loss_lpips = self.calc_loss(generated_images, real_images_batch, image_name,
                                                               self.G, use_ball_holder, w_pivot)

                self.optimizer.zero_grad()

                if loss_lpips <= hyperparameters.LPIPS_value_threshold:
                    break

                loss.backward()
                self.optimizer.step()

                use_ball_holder = global_config.training_step % hyperparameters.locality_regularization_interval == 0

                global_config.training_step += 1

            self.image_counter += 1

            torch.save(self.G,
                       f'{paths_config.checkpoints_dir}/model_{global_config.run_name}_{image_name}.pt')
