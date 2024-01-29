import click
import torch
from PIL import Image
import numpy as np
import os


def rgb2hsv(color: torch.Tensor) -> torch.Tensor:
    color = color.to(torch.float) / 255
    flat = False
    if len(color.shape) == 1:
        color = color[None, :, None, None]
        flat = True
    max_c = color.max(axis=1).values
    min_c = color.min(axis=1).values
    delta_c = max_c - min_c
    is_eq = max_c == min_c
    bad = ~is_eq
    is_r = (max_c == color[:, 0]) & bad
    bad &= ~is_r
    is_g = (max_c == color[:, 1]) & bad
    bad &= ~is_g
    is_b = (max_c == color[:, 2]) & bad
    color[:, 0][is_eq] = 0
    color[:, 0][is_r] = (60 * ((color[:, 1][is_r] - color[:, 2][is_r]) / delta_c[is_r]) + 360) % 360
    color[:, 0][is_g] = 60 * ((color[:, 2][is_g] - color[:, 0][is_g]) / delta_c[is_g]) + 120
    color[:, 0][is_b] = 60 * ((color[:, 0][is_b] - color[:, 1][is_b]) / delta_c[is_b]) + 240
    color[:, 1] = torch.where(max_c == 0, 0, 1 - min_c / max_c)
    color[:, 2] = max_c
    return color.flatten() if flat else color


def hsv2rgb(color: torch.Tensor) -> torch:
    flat = False
    if len(color.shape) == 1:
        color = color[None, :, None, None]
        flat = True
    h60 = (color[:, 0] / 60)
    i = h60.to(torch.uint8) % 6
    f = h60 - i
    v = color[:, 2]
    l = v * (1 - color[:, 1])
    m = v * (1 - f * color[:, 1])
    n = v * (1 - (1 - f) * color[:, 1])
    results = [[v, n, l], [m, v, l], [l, v, n], [l, m, v], [n, l, v], [v, l, m]]
    color = color.permute(0, 2, 3, 1)
    for j in range(6):
        mask = i == j
        r, g, b = results[j]
        color[mask] = torch.stack([r[mask], g[mask], b[mask]], axis=1)
    color = color.permute(0, 3, 1, 2)*255
    return color.flatten() if flat else color


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
    if type == 'hsv':
        imgR = hsv2rgb(imgR)
    imgR = imgR.permute(0, 2, 3, 1).to(torch.uint8)[0].cpu().numpy()
    Image.fromarray(imgR, 'RGB').save(f'{outdir}/replaced.png')


if __name__ == '__main__':
    main()
