import click
import torch
from PIL import Image
import numpy as np
import os


def replaceColor(imgs, imgs_new_color, color, epsilon):
    assert imgs.shape == imgs_new_color.shape, f'Shape {imgs.shape} not equal {imgs_new_color.shape}'
    assert color.shape == torch.Size([3])
    cond = getMask(imgs, color, epsilon)
    imgs[cond] = imgs_new_color[cond]
    return imgs


def pasteColor(imgs_color, imgs, color, epsilon):
    assert imgs_color.shape == imgs.shape, f'Shape {imgs_color.shape} not equal {imgs.shape}'
    assert color.shape == torch.Size([3])
    cond = getMask(imgs_color, color, epsilon)
    imgs[cond] = imgs_color[cond]
    return imgs


def maskColor(imgs, color, epsilon):
    mask = ~getMask(imgs, color, epsilon)
    return imgs * mask


def getMask(imgs, color, epsilon):
    imgs = imgs.permute(0, 2, 3, 1)
    cond = (imgs >= (color - epsilon)) & (imgs <= (color + epsilon))
    return cond.all(axis=3)[:, None, :, :].repeat(1, imgs.shape[-1], 1, 1)


def loadImg(path):
    return torch.from_numpy(np.array(Image.open(path).convert('RGB'))).permute(2, 0, 1)


@click.command()
@click.option('--img1', 'img1_path', type=str)
@click.option('--img2', 'img2_path', type=str)
@click.option('--device', 'device_name', default='cuda', type=torch.device)
@click.option('--epsilon')
@click.option('--color', help='a color list, value of composante in float range [0.,255.]')
@click.option('--outdir', default='out')
def main(img1_path, img2_path, device_name, epsilon, color, outdir):
    os.makedirs(outdir, exist_ok=True)
    device = torch.device(device_name)
    img1 = loadImg(img1_path).to(device)[None]
    img2 = loadImg(img2_path).to(device)[None]
    color = color[1:-1].split(',')
    for i in range(len(color)): color[i] = float(color[i])
    color = torch.tensor(color).to(device)
    try:
        epsilon = float(epsilon)
    except ValueError:
        epsilon = epsilon[1:-1].split(',')
        for i in range(len(epsilon)): epsilon[i] = float(epsilon[i])
        epsilon = torch.tensor(epsilon).to(device)
    imgR = replaceColor(img1, img2, color, epsilon)
    imgR = imgR.permute(0, 2, 3, 1).to(torch.uint8)[0].cpu().numpy()
    Image.fromarray(imgR, 'RGB').save(f'{outdir}/replaced.png')


if __name__ == '__main__':
    main()
