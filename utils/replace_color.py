import click
import torch
from PIL import Image
from torchvision.transforms import transforms

def replaceColor(imgs,imgs_new_color,color,epsilon):
    assert imgs.shape == imgs_new_color.shape, f'Shape {imgs.shape} not equal {imgs_new_color.shape}'
    assert color.shape == torch.Size([3])
    imgs = imgs.permute(0,2,3,1)
    cond = (imgs>=(color-epsilon)) & (imgs<=(color+epsilon))
    cond = cond[:,:,:,0] & cond[:,:,:,1] & cond[:,:,:,2]
    imgs[cond]=imgs_new_color.permute(0,2,3,1)[cond]
    imgs = imgs.permute(0,3,1,2)
    return imgs

def pasteColor(imgs_color, imgs, color, epsilon):
    assert imgs_color.shape == imgs.shape, f'Shape {imgs_color.shape} not equal {imgs.shape}'
    assert color.shape == torch.Size([3])
    imgs_color = imgs_color.permute(0,2,3,1)
    cond = (imgs_color>=(color-epsilon)) & (imgs_color<=(color+epsilon))
    cond = cond[:,:,:,0] & cond[:,:,:,1] & cond[:,:,:,2]
    imgs = imgs.permute(0,2,3,1)
    imgs[cond]=imgs_color[cond]
    imgs = imgs.permute(0,3,1,2)
    imgs_color = imgs_color.permute(0,3,1,2)
    return imgs

def loadImg(path):
    img = Image.open(path).convert('RGB')
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])(img)


@click.command()
@click.option('--img1','img1_path', type=str)
@click.option('--img2','img2_path', type=str)
@click.option('--device','device_name',default='cuda', type=torch.device)
@click.option('--epsilon')
@click.option('--color', help='a color list, value of composante in float range (-1)-(1)')
@click.option('--outdir', default='out')
def main(img1_path, img2_path, device_name, epsilon, color, outdir):
    device = torch.device(device_name)
    img1 = loadImg(img1_path).to(device)[None,:,:,:]
    img2 = loadImg(img2_path).to(device)[None,:,:,:]
    color = color[1:-1].split(',')
    for i in range(len(color)): color[i]=float(color[i])
    color = torch.tensor(color).to(device)
    try: epsilon=float(epsilon)
    except:
        epsilon=epsilon[1:-1].split(',')
        for i in range(len(epsilon)): epsilon[i]=float(epsilon[i])
        epsilon = torch.tensor(epsilon).to(device)
    imgR = pasteColor(img1, img2, color, epsilon)
    imgR = (imgR + 1) * (255/2)
    imgR = imgR.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    Image.fromarray(imgR, 'RGB').save(f'{outdir}/replaced.png')

if __name__ == '__main__':
    main()