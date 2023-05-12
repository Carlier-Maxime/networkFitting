import pickle
import functools
import torch
from configs import paths_config


def toogle_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def load_tuned_G(run_id, type, device):
    new_G_path = f'{paths_config.checkpoints_dir}/model_{run_id}_{type}.pt'
    with open(new_G_path, 'rb') as f:
        new_G = torch.load(f).to(device).eval()
    new_G = new_G.float()
    toogle_grad(new_G, False)
    return new_G


def load_network(path, device:torch.device):
    G = None
    with open(path, 'rb') as f:
        if path.endswith(".pkl"):
            G = pickle.load(f)['G_ema'].to(device).eval()
            G = G.float()
        elif path.endswith(".pt"):
            G = torch.load(f).to(device)
    return G
