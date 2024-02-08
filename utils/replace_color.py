import time

import click
import torch
from PIL import Image
import numpy as np
import os
import torch.nn.functional as F


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
    color = color.permute(0, 3, 1, 2) * 255
    return color.flatten() if flat else color


def replaceColor(imgs, imgs_new_color, color, epsilon, grow_size=1):
    assert imgs.shape == imgs_new_color.shape, f'Shape {imgs.shape} not equal {imgs_new_color.shape}'
    assert color.shape == torch.Size([3])
    cond = getMask(imgs, color, epsilon, grow_size)
    imgs.permute(0, 2, 3, 1)[cond] = imgs_new_color.permute(0, 2, 3, 1)[cond]
    return imgs


def pasteColor(imgs_color, imgs, color, epsilon, grow_size=1):
    assert imgs_color.shape == imgs.shape, f'Shape {imgs_color.shape} not equal {imgs.shape}'
    assert color.shape == torch.Size([3])
    cond = getMask(imgs_color, color, epsilon, grow_size)
    imgs.permute(0, 2, 3, 1)[cond] = imgs_color.permute(0, 2, 3, 1)[cond]
    return imgs


def maskColor(imgs, color, epsilon, grow_size=1):
    mask = ~getMask(imgs, color, epsilon, grow_size)
    return imgs * mask


def save_mask_stain(maskStains):
    mask_color = ((maskStains / maskStains.max()) * (256 ** 3 - 1)).repeat(3, 1, 1)
    mask_color[0] %= 256
    mask_color[1] = (mask_color[1] / 256) % 256
    mask_color[2] = (mask_color[2] / 256) / 256
    mask_color = mask_color.clip(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    Image.fromarray(mask_color).save(f'{global_outdir}/mask_stain.png')


def getCentersOfStain(masks=torch.tensor([[[0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0]]], dtype=torch.bool, device='cuda')):
    start_time = time.time()
    indices = torch.where(masks)
    indices = torch.stack((indices[1], indices[2])).split(indices[0].unique_consecutive(return_counts=True)[1].tolist(), dim=1)
    markers = masks.long()
    for i in range(len(indices)):
        unique = indices[i][0].unique_consecutive(return_counts=True)
        columns = torch.where(indices[i][0].eq(unique[0][..., None]))[1].split(unique[1].tolist())
        prev_column = None
        for j in range(len(columns)):
            column = indices[i][1, columns[j]]
            split_indices = torch.where((column[1:] - column[:-1]) > 2)[0] + 1
            start_indices = torch.cat((torch.tensor([0], device=column.device), split_indices))
            end_indices = torch.cat([split_indices, torch.tensor([column.shape[0]], device=column.device)])
            ranges = torch.arange(column.shape[0], device=column.device)[:, None]
            values = ((ranges.ge(start_indices) & ranges.lt(end_indices)) * torch.arange(len(end_indices), device=column.device)).sum(axis=1) + (j*indices[i].shape[1]) + 2
            y = unique[0][j]
            y_sub_1 = unique[0][j-1]
            markers[i, y, column] = values
            if prev_column is not None:
                same = masks[i, y_sub_1, column] & masks[i, y, column]
                pairs = torch.stack((values[same], markers[i, y_sub_1, column[same]]), dim=1).unique(dim=0)
                for k in range(len(pairs)):
                    markers[markers == pairs[k, 0]] = pairs[k, 1]
                    pairs[pairs[:, 0] == pairs[k, 0], 0] = pairs[k, 1]
            prev_column = column
    if global_save_mask: save_mask_stain(markers)
    markers_id = markers.unique()
    markers_id = markers_id[markers_id > 0]
    where = torch.where(markers.eq(markers_id[..., None, None]))
    markers = torch.stack([where[1], where[2]]).split(where[0].unique(return_counts=True)[1].tolist(), dim=1)
    centers = torch.stack([marker.float().mean(axis=1) for marker in markers])
    print(time.time() - start_time)
    return centers


def getMask(imgs, color, epsilon, grow_size=1):
    imgs = imgs.permute(0, 2, 3, 1)
    cond = ((imgs >= (color - epsilon)) & (imgs <= (color + epsilon))).all(axis=3)
    if global_save_ccs:
        centers = getCentersOfStain(cond)
        print(f"Number of color stain detected : {len(centers)}")
        np.save(f"{global_outdir}/centers.npy", centers.cpu().numpy())
    if grow_size > 1:
        kernel = torch.ones(cond.shape[0], 1, grow_size, grow_size, device=cond.device)
        cond = torch.gt(F.conv2d(cond[:, None].to(torch.float), kernel, padding='same'), 0)[:, 0]
    if global_save_mask:
        Image.fromarray((cond.to(torch.uint8) * 255)[0].cpu().numpy()).save(f"{global_outdir}/mask.png")
    return cond


def loadImg(path):
    return torch.from_numpy(np.array(Image.open(path).convert('RGB'))).permute(2, 0, 1)


def eraseColor(imgs, color, epsilon, grow_size=1, erase_size=5):
    assert color.shape == torch.Size([3])
    assert erase_size > 1
    cond = getMask(imgs, color, epsilon, grow_size)
    while cond.any():
        kernel = torch.ones(1, 1, erase_size, erase_size, device=cond.device)
        nbPixelsIgnored = F.conv2d(cond[:, None].to(torch.float), kernel, padding='same')[:, 0]
        sumPixelsKernel = F.conv2d(imgs * ~cond, kernel.repeat((imgs.shape[1], 1, 1, 1)), padding='same', groups=imgs.shape[1])
        div = erase_size ** 2 - nbPixelsIgnored
        nextCond = div == 0
        cond = cond ^ nextCond
        imgs.permute(0, 2, 3, 1)[cond] = (sumPixelsKernel.permute(0, 2, 3, 1)[cond].permute(1, 0) / div[cond]).permute(1, 0)
        cond = nextCond
    return imgs


global_outdir = None
global_save_ccs = None
global_save_mask = None


@click.command()
@click.option('--img1', 'img1_path', type=str)
@click.option('--img2', 'img2_path', type=str)
@click.option('--mode', help='mode used for change color', type=click.Choice(['mask', 'replace', 'paste', 'erase']), default='replace')
@click.option('--device', 'device_name', default='cuda', type=torch.device)
@click.option('--epsilon', metavar='[color|float]')
@click.option('--color', help='a color list, value of composante in float range [0.,255.]', metavar='color', type=str)
@click.option('--outdir', default='out')
@click.option('--type', 'type_c', help='a type of color data', type=click.Choice(['rgb', 'hsv']), default='rgb')
@click.option('--grow', 'grow_size', help='dilating a zone of specific color for prevent outline mistake', type=click.IntRange(min=1), default=1)
@click.option('--erase_size', help='size of kernel used for calcul average of surrounding pixels', type=click.IntRange(min=2), default=5)
@click.option('--save-ccs', help='Save a center of color stain to a numpy file', is_flag=True, default=False)
@click.option('--save-mask', help='Save a mask to a PNG', is_flag=True, default=False)
def main(img1_path, img2_path, mode, device_name, epsilon, color, outdir, type_c, grow_size, erase_size, save_ccs, save_mask):
    global global_outdir, global_save_ccs, global_save_mask
    global_outdir = outdir
    global_save_ccs = save_ccs
    global_save_mask = save_mask
    os.makedirs(outdir, exist_ok=True)
    device = torch.device(device_name)
    img1 = loadImg(img1_path).to(device)[None]
    img2 = loadImg(img2_path).to(device)[None] if mode in ['replace', 'paste'] else None
    if type_c == 'hsv':
        img1 = rgb2hsv(img1)
        img2 = None if img2 is None else rgb2hsv(img2)
    color = color[1:-1].split(',')
    for i in range(len(color)): color[i] = float(color[i])
    color = torch.tensor(color).to(device)
    try:
        epsilon = float(epsilon)
    except ValueError:
        epsilon = epsilon[1:-1].split(',')
        for i in range(len(epsilon)): epsilon[i] = float(epsilon[i])
        epsilon = torch.tensor(epsilon).to(device)
    if mode == 'mask':
        imgR = maskColor(img1, color, epsilon, grow_size)
    elif mode == 'replace':
        imgR = replaceColor(img1, img2, color, epsilon, grow_size)
    elif mode == 'paste':
        imgR = pasteColor(img1, img2, color, epsilon, grow_size)
    elif mode == 'erase':
        imgR = eraseColor(img1, color, epsilon, grow_size, erase_size)
    else:
        raise ValueError('mode unknown : ' + mode)
    if type_c == 'hsv': imgR = hsv2rgb(imgR)
    imgR = imgR.permute(0, 2, 3, 1).to(torch.uint8)[0].cpu().numpy()
    Image.fromarray(imgR, 'RGB').save(f'{outdir}/replaced.png')


if __name__ == '__main__':
    main()
