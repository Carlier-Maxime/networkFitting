import os

import torch
from tqdm import tqdm
from configs import paths_config, hyperparameters, global_config
from coaches.base_coach import BaseCoach

class MultiIDCoach(BaseCoach):

    def __init__(self, data_loader):
        super().__init__(data_loader)

    def train(self):
        self.G.synthesis.train()
        self.G.mapping.train()

        w_path_dir = f'{paths_config.embedding_base_dir}/{paths_config.input_data_id}'
        os.makedirs(w_path_dir, exist_ok=True)
        os.makedirs(f'{w_path_dir}/{paths_config.pti_results_keyword}', exist_ok=True)

        use_ball_holder = True
        w_pivots = []
        images = []

        for fname, image in self.data_loader:
            if self.image_counter >= hyperparameters.max_images_to_invert:
                break

            image_name = fname[0]
            embedding_dir = f'{w_path_dir}/{paths_config.pti_results_keyword}/{image_name}'
            os.makedirs(embedding_dir, exist_ok=True)

            w_pivot = self.get_inversion(w_path_dir, image_name, image)
            w_pivots.append(w_pivot)
            images.append((image_name, image))
            self.image_counter += 1

        for i in tqdm(range(hyperparameters.max_pti_steps)):
            self.image_counter = 0

            for data, w_pivot in zip(images, w_pivots):
                image_name, image = data

                if self.image_counter >= hyperparameters.max_images_to_invert:
                    break

                real_images_batch = image.to(global_config.device)

                generated_images = self.forward(w_pivot[0].repeat(1,self.G.num_ws,1))
                loss, l2_loss_val, loss_lpips = self.calc_loss(generated_images, real_images_batch, image_name,
                                      self.G, use_ball_holder, w_pivot)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                use_ball_holder = global_config.training_step % hyperparameters.locality_regularization_interval == 0

                global_config.training_step += 1
                self.image_counter += 1

        torch.save(self.G,
                   f'{paths_config.checkpoints_dir}/model_{global_config.run_name}_multi_id.pt')
