import os

import click
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

import sys
sys.path.append(".")
from stylegan_xl import dnnlib


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
        color[mask] = torch.stack([r[mask], g[mask], b[mask]], dim=1)
    color = color.permute(0, 3, 1, 2) * 255
    return color.flatten() if flat else color


def replaceOrPasteColor(imgs1, imgs2, color, epsilon, grow_size=1, paste: bool = False):
    assert imgs1.shape == imgs2.shape, f'Shape {imgs1.shape} not equal {imgs2.shape}'
    assert color.shape == torch.Size([3])
    cond = getMask(imgs1, color, epsilon, grow_size)
    (imgs2 if paste else imgs1).permute(0, 2, 3, 1)[cond] = (imgs1 if paste else imgs2).permute(0, 2, 3, 1)[cond]
    return imgs1


def maskColor(imgs, color, epsilon, grow_size=1):
    mask = ~getMask(imgs, color, epsilon, grow_size)
    return imgs * mask


def save_mask_stain(maskStains, min_light=32):
    color_interval = (256-min_light)
    mask_color = ((maskStains / maskStains.max()) * (color_interval ** 3 - 1)).repeat(3, 1, 1, 1)
    mask_color[0] = (mask_color[0] % color_interval) + min_light
    mask_color[1] = ((mask_color[1] / color_interval) % color_interval) + min_light
    mask_color[2] = ((mask_color[2] / color_interval) / color_interval) + min_light
    mask_color[mask_color == min_light] = 0
    mask_color = mask_color.clip(0, 255).to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()
    for i in range(len(mask_color)): Image.fromarray(mask_color[i]).save(f'{opts.outdir_mask_stained}/mask_stain{"" if opts.current_index is None else opts.current_index+i}.png')


def getCentersOfStain(masks: torch.Tensor):
    markers = masks.long()
    markers[masks] = torch.nn.ConstantPad2d((1, 1, 0, 0), False)(masks).unique_consecutive(return_inverse=True)[1][:, :, 1:-1][masks]
    sames = masks[:, :-1] & masks[:, 1:]
    pairs = torch.stack((markers[:, :-1][sames], markers[:, 1:][sames]), dim=1).unique(dim=0)
    for k in range(len(pairs)): markers[markers == pairs[k, 0]] = pairs[k, 1]
    if opts.save_mask: save_mask_stain(markers)
    markers_id = markers.unique()
    markers_id = markers_id[markers_id > 0]
    where = torch.where(markers.eq(markers_id[..., None, None, None]))
    nb_pixels = where[0].unique(return_counts=True)[1]
    markers = torch.stack([where[2], where[3]]).split(nb_pixels.tolist(), dim=1)
    centers = torch.stack([marker.float().mean(axis=1) for marker in markers])
    return centers.split(where[1][nb_pixels.cumsum(dim=0).sub(1)].unique(return_counts=True)[1].tolist())


def getMask(imgs, color, epsilon, grow_size=1):
    imgs = imgs.permute(0, 2, 3, 1)
    cond = ((imgs >= (color - epsilon)) & (imgs <= (color + epsilon))).all(axis=3)
    if opts.save_ccs:
        centers = getCentersOfStain(cond)
        for i in range(len(centers)):
            save_index = "" if opts.current_index is None else opts.current_index+i
            print(f"Number of color stain detected in image {save_index} : {len(centers[i])}")
            np.save(f"{opts.outdir_ccs}/centers{save_index}.npy", centers[i].cpu().numpy())
    if grow_size > 1:
        kernel = torch.ones(cond.shape[0], 1, grow_size, grow_size, device=cond.device)
        cond = torch.gt(F.conv2d(cond[:, None].to(torch.float), kernel, padding='same'), 0)[:, 0]
    if opts.save_mask:
        masks = (cond.to(torch.uint8) * 255).cpu().numpy()
        for i in range(len(masks)): Image.fromarray(masks[i]).save(f"{opts.outdir_mask}/mask{'' if opts.current_index is None else opts.current_index+i}.png")
    return cond


def loadImg(path):
    return torch.from_numpy(np.array(Image.open(path).convert('RGB'))).permute(2, 0, 1)


def eraseColor(imgs, color, epsilon, grow_size=1, erase_size=5):
    assert color.shape == torch.Size([3])
    assert erase_size > 2
    cond = getMask(imgs, color, epsilon, grow_size)
    if imgs.dtype == torch.uint8: imgs = imgs.to(torch.float)
    kernel = torch.linspace(-1, 1, erase_size, device=cond.device).abs().mul(-1).add(1).repeat(erase_size, 1)
    kernel = torch.stack((kernel, kernel.permute(1, 0)), dim=2).mean(dim=2)[None, None]
    max_div = kernel.sum()
    while cond.any():
        nbPixelsIgnored = F.conv2d(cond[:, None].to(torch.float), kernel, padding='same')[:, 0]
        sumPixelsKernel = F.conv2d(imgs * ~cond, kernel.repeat((imgs.shape[1], 1, 1, 1)), padding='same', groups=imgs.shape[1])
        div = max_div - nbPixelsIgnored
        nextCond = div < erase_size >> 1
        cond = cond ^ nextCond
        imgs.permute(0, 2, 3, 1)[cond] = (sumPixelsKernel.permute(0, 2, 3, 1)[cond].permute(1, 0) / div[cond]).permute(1, 0)
        cond = nextCond
    return imgs


def process(mode, type_c, img1, img2, color, epsilon, grow_size, erase_size):
    if mode == 'mask':
        imgR = maskColor(img1, color, epsilon, grow_size)
    elif mode == 'replace':
        imgR = replaceOrPasteColor(img1, img2, color, epsilon, grow_size)
    elif mode == 'paste':
        imgR = replaceOrPasteColor(img1, img2, color, epsilon, grow_size, paste=True)
    elif mode == 'erase':
        imgR = eraseColor(img1, color, epsilon, grow_size, erase_size)
    else:
        raise ValueError('mode unknown : ' + mode)
    return imgR if type_c == 'RGB' else hsv2rgb(imgR)


def videoProcess(path1: str, path2, color: torch.Tensor, epsilon: torch.Tensor, grow_size: int = 3, erase_size: int = 5, batch: int = 1, type_c='RGB', mode='erase'):
    from ImagesDataset import ImagesByVideoDataset
    data = DataLoader(ImagesByVideoDataset(path1, type_c), batch_size=batch, shuffle=False)
    data2 = data if path2 is None else DataLoader(ImagesByVideoDataset(path2, type_c), batch_size=batch, shuffle=False)
    frame = 0
    opts.current_index = 0
    for imgs1, imgs2 in zip(data, data2):
        if not imgs1.numel() or not imgs2.numel(): continue
        imgsR = process(mode, type_c, imgs1, imgs2, color, epsilon, grow_size, erase_size)
        imgsR = imgsR.permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy()
        for img in imgsR:
            Image.fromarray(img, 'RGB').save(f'{opts.outdir_result}/frame{frame}.png')
            frame += 1
        opts.current_index = frame


def imageProcess(path1, path2, mode, color, epsilon, grow_size, erase_size, type_c):
    img1 = loadImg(path1).to(color.device)[None]
    img2 = loadImg(path2).to(color.device)[None] if mode in ['replace', 'paste'] else None
    if type_c == 'HSV':
        img1 = rgb2hsv(img1)
        img2 = None if img2 is None else rgb2hsv(img2)
    imgR = process(mode, type_c, img1, img2, color, epsilon, grow_size, erase_size)
    imgR = imgR.permute(0, 2, 3, 1).to(torch.uint8)[0].cpu().numpy()
    Image.fromarray(imgR, 'RGB').save(f'{opts.outdir}/replaced.png')


def str2tensor(_ctx, _param, value):
    value = value[1:-1].split(',')
    for i in range(len(value)): value[i] = float(value[i])
    return torch.tensor(value)


def str2floatOrTensor(_ctx, _param, value):
    try:
        return float(value)
    except ValueError:
        return str2tensor(_ctx, _param, value)


global opts


@click.command()
@click.option('--path1', type=str)
@click.option('--path2', type=str)
@click.option('--mode', help='mode used for change color', type=click.Choice(['mask', 'replace', 'paste', 'erase']), default='replace')
@click.option('--device', default='cuda', type=torch.device)
@click.option('--epsilon', metavar='[color|float]', callback=str2floatOrTensor)
@click.option('--color', help='a color list, value of composante in float range [0.,255.]', metavar='color', type=str, callback=str2tensor)
@click.option('--outdir', default='out')
@click.option('--type', 'type_c', help='a type of color data', type=click.Choice(['rgb', 'hsv']), default='rgb')
@click.option('--grow', 'grow_size', help='dilating a zone of specific color for prevent outline mistake', type=click.IntRange(min=1), default=1)
@click.option('--erase_size', help='size of kernel used for calcul average of surrounding pixels', type=click.IntRange(min=2), default=5)
@click.option('--save-ccs', help='Save a center of color stain to a numpy file', is_flag=True, default=False)
@click.option('--save-mask', help='Save a mask to a PNG', is_flag=True, default=False)
def main(**kwargs):
    global opts
    opts = dnnlib.EasyDict(kwargs)
    opts.outdir = opts.outdir+'/'+opts.path1.split("/")[-1].split(".")[0]
    opts.outdir_ccs = opts.outdir+'/ccs'
    opts.outdir_mask = opts.outdir+'/mask'
    opts.outdir_mask_stained = opts.outdir+'/mask_stained'
    opts.outdir_result = opts.outdir+'/result'
    opts.device = torch.device(opts.device)
    for directory in [opts.outdir, opts.outdir_mask, opts.outdir_mask_stained, opts.outdir_ccs, opts.outdir_result]: os.makedirs(directory, exist_ok=True)
    opts.color = opts.color.to(opts.device)
    opts.epsilon = opts.epsilon.to(opts.device)
    opts.type_c = opts.type_c.upper()
    if opts.path1.split(".")[-1].lower() not in ['png', 'jpg', 'jpeg']:
        videoProcess(opts.path1, opts.path2, opts.color, opts.epsilon, opts.grow_size, opts.erase_size, 1, opts.type_c, opts.mode)
    else:
        imageProcess(opts.path1, opts.path2, opts.mode, opts.color, opts.epsilon, opts.grow_size, opts.erase_size, opts.type_c)


if __name__ == '__main__':
    main()
