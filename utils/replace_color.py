import click
import torch
from PIL import Image
import numpy as np
import os


def rgb2hsv(rgb: torch.Tensor) -> torch.Tensor:
    rgb = rgb.to(torch.float) / 255
    hsv = torch.zeros(3, device=rgb.device, dtype=rgb.dtype)
    max_c = rgb.max()
    min_c = rgb.min()
    delta_c = max_c - min_c
    if max_c == min_c: pass
    elif max_c == rgb[0]: hsv[0] = 60 * ((rgb[1] - rgb[2]) / delta_c) + 360 % 360
    elif max_c == rgb[1]: hsv[0] = 60 * ((rgb[2] - rgb[0]) / delta_c) + 120
    elif max_c == rgb[2]: hsv[0] = 60 * ((rgb[0] - rgb[1]) / delta_c) + 240
    hsv[1] = 0 if max == 0 else (1 - min_c / max_c)
    hsv[2] = max_c
    return hsv


def hsv2rgb(hsv: torch.Tensor) -> torch:
    h60 = hsv[0] / 60
    i = int(h60) % 6
    f = h60 - i
    v = hsv[2]
    l = v * (1 - hsv[1])
    m = v * (1 - f * hsv[1])
    n = v * (1 - (1 - f) * hsv[1])
    if i == 0: rgb = [v, n, l]
    elif i == 1: rgb = [m, v, l]
    elif i == 2: rgb = [l, v, n]
    elif i == 3: rgb = [l, m, v]
    elif i == 4: rgb = [n, l, v]
    elif i == 5: rgb = [v, l, m]
    else: raise ValueError()
    return torch.tensor(rgb, device=hsv.device)*255


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
@click.option('--epsilon', metavar='[color|float]')
@click.option('--color', help='a color list, value of composante in float range [0.,255.]', metavar='color', type=str)
@click.option('--outdir', default='out')
@click.option('--type', help='a type of color data', type=click.Choice(['rgb', 'hsv']), default='rgb')
def main(img1_path, img2_path, device_name, epsilon, color, outdir, type):
    os.makedirs(outdir, exist_ok=True)
    device = torch.device(device_name)
    img1 = loadImg(img1_path).to(device)[None]
    img2 = loadImg(img2_path).to(device)[None]
    if type == 'hsv':
        img1 = rgb2hsv(img1)
        img2 = rgb2hsv(img2)
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
