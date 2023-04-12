import click, torch, sys, cv2, PIL, numpy as np, os, copy, imageio, dill
from time import perf_counter
from tqdm import tqdm,trange
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
sys.path.insert(1, 'stylegan-xl')
import dnnlib, legacy
from torch_utils.ops import filtered_lrelu, bias_act
from torch_utils import gen_utils
from run_inversion import project, space_regularizer_loss
from metrics import metric_utils

from utils.ImagesDataset import ImagesDataset
from coaches.multi_id_coach import MultiIDCoach

def loadNetwork(network_pkl:str, device:torch.device, verbose:bool=True):
    if verbose: print('Loading networks from "%s"...' % network_pkl)
    device = torch.device(device)
    with dnnlib.util.open_url(network_pkl) as fp:
        G = legacy.load_network_pkl(fp)['G_ema'].to(device) # type: ignore
    return G

def getImagesFromVideo(filename:str, ips:int, device:torch.device, verbose:bool=True, dezired_size:int=-1):
    video=cv2.VideoCapture(filename)
    steps=(video.get(cv2.CAP_PROP_FRAME_COUNT)/video.get(cv2.CAP_PROP_FPS))*ips
    frame_step = int(video.get(cv2.CAP_PROP_FPS)/ips)
    if frame_step==0:
        frame_step=1
    images = []
    read_counter=0
    for i in trange(1,int(steps+1), desc='Loading images', unit='image', disable=(not verbose)):
        target_pil = None
        ret,cv2_im = video.read()
        while (read_counter % frame_step != 0):
            ret,cv2_im = video.read()
            if not ret:
                break
            read_counter+=1
        if not ret:
            break
        converted = cv2.cvtColor(cv2_im,cv2.COLOR_BGR2RGB)
        target_pil = PIL.Image.fromarray(converted)
        read_counter+=1
        w, h = target_pil.size
        s = min(w, h)
        target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        if dezired_size<=0: dezired_size=s
        target_pil = target_pil.resize((dezired_size, dezired_size), PIL.Image.LANCZOS)
        img_np = np.array(target_pil, dtype=np.uint8)
        images.append(torch.tensor(img_np.transpose([2, 0, 1]), device=device))
    return images

def initPlugins(verbose:bool=True):
    verbosity = 'brief' if verbose else 'none'
    bias_act._init(verbosity)
    filtered_lrelu._init(verbosity)

def calculLatents(
    G,
    dataloader:DataLoader,
    device:torch.device,
    first_inv_steps:int=1000,
    inv_steps:int=100,
    w_init_path:str=None,
    save_latent:bool=False,
    save_video_latent:bool=False,
    outdir:str='out',
    ips:int=60
):
    w_pivots = []
    if w_init_path:
        w_pivots.append(torch.from_numpy(np.load(w_init_path)['w'])[0].to(device))
    w_imgs = []
    wrimgs = []
    for name,img in tqdm(dataloader, desc='Calcul latents', unit='image'):
        imgs, w_pivot = project(
            G,
            target=img[0], # pylint: disable=not-callable
            num_steps=(inv_steps if len(w_pivots)>0 else first_inv_steps),
            device=device,
            verbose=True,
            noise_mode='const',
            w_start_pivot=(w_pivots[-1] if len(w_pivots)>0 else None)
        )
        w_pivots.append(w_pivot)
        if save_latent:
            np.savez(f'{outdir}/latent{name[0]}.npz', w=w_pivot.unsqueeze(0).cpu().detach().numpy())
        if save_video_latent:
            w_imgs += imgs
            synth_image = G.synthesis(w_pivot.repeat(1, G.num_ws, 1))
            synth_image = (synth_image + 1) * (255/2)
            synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
            wrimgs.append(synth_image)
    
    if save_video_latent:
        video = imageio.get_writer(f'{outdir}/optiLatent.mp4', mode='I', fps=60, codec='libx264', bitrate='16M')
        for synth_image in w_imgs:
            video.append_data(np.array(synth_image))
        video.close()
        video = imageio.get_writer(f'{outdir}/resultLatent.mp4', mode='I', fps=ips, codec='libx264', bitrate='16M')
        for synth_image in wrimgs:
            video.append_data(np.array(synth_image))
        video.close()
    return w_pivots

def pti_multiple_targets(
    G,
    w_pivots,
    dataloader:DataLoader,
    device: torch.device,
    num_steps=350,
    learning_rate = 3e-4,
    noise_mode="const",
    verbose = False,
    seed=64,
    save_video = False,
    outdir:str = 'out',
    disable_gradient_reg_loss:bool = False
):
    G_pti = copy.deepcopy(G).train().requires_grad_(True).to(device)
    w_seed = gen_utils.get_w_from_seed(G, 1, device, seed=seed)

    # Load VGG16 feature detector.
    vgg16_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/vgg16.pkl'
    vgg16 = metric_utils.get_feature_detector(vgg16_url, device=device)

    # l2 criterion
    l2_criterion = torch.nn.MSELoss(reduction='mean')

    # initalize optimizer
    optimizer = torch.optim.Adam(G_pti.parameters(), lr=learning_rate)

    # run optimization loop
    seed_images = []
    target_images = []
    w_pivots[0].requires_grad_(False)
    i=0
    iterator = iter(dataloader)
    for step in (pbar := trange(1,num_steps+1, desc='Optimization PTI', unit='step', disable=(not verbose))):
        if save_video:
            synth_images = G_pti.synthesis(w_seed, noise_mode=noise_mode)
            synth_images = (synth_images + 1) * (255/2)
            synth_images_np = synth_images.clone().detach().permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
            seed_images.append(synth_images_np)

            synth_images = G_pti.synthesis(w_pivots[0][0].repeat(1,G.num_ws,1), noise_mode=noise_mode)
            synth_images = (synth_images + 1) * (255/2)
            synth_images_np = synth_images.clone().detach().permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
            target_images.append(synth_images_np)
        
        # Features for target image.
        try: target_image = next(iterator)[1][0].unsqueeze(0).to(device).to(torch.float32)
        except:
            iterator = iter(dataloader)
            target_image = next(iterator)[1][0].unsqueeze(0).to(device).to(torch.float32)
        if target_image.shape[2] > 256:
            target_image = F.interpolate(target_image, size=(256, 256), mode='area')
        target_features = vgg16(target_image, resize_images=False, return_lpips=True)

        # Synth images from opt_w.
        w_pivots[i%len(w_pivots)].requires_grad_(False)
        synth_images = G_pti.synthesis(w_pivots[i%len(w_pivots)][0].repeat(1,G.num_ws,1), noise_mode=noise_mode)
        synth_images = (synth_images + 1) * (255/2)

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        if synth_images.shape[2] > 256:
            synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')

        # LPIPS loss
        synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
        lpips_loss = (target_features - synth_features).square().sum()

        # MSE loss
        mse_loss = l2_criterion(target_image, synth_images)

        # space regularizer
        reg_loss = space_regularizer_loss(G_pti, G, w_pivots[i%len(w_pivots)], vgg16, disable_gradient=disable_gradient_reg_loss).to(device)

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss = mse_loss + lpips_loss + reg_loss
        loss.backward()
        optimizer.step()
        i+=1

        if verbose:
            pbar.set_postfix_str(f'loss: {float(loss):<5.2f}, lpips: {float(lpips_loss):<5.2f}, mse: {float(mse_loss):<5.2f}, reg: {float(reg_loss):<5.2f}')

    if save_video:
        print (f'Saving network fitting progress video "{outdir}/fitting_seed.mp4" and "{outdir}/fitting_target.mp4"')
        video = imageio.get_writer(f'{outdir}/fitting_seed.mp4', mode='I', fps=60, codec='libx264', bitrate='16M')
        for synth_image in seed_images:
            video.append_data(synth_image)
        video.close()
        video = imageio.get_writer(f'{outdir}/fitting_target.mp4', mode='I', fps=60, codec='libx264', bitrate='16M')
        for synth_image in target_images:
            video.append_data(synth_image)
        video.close()

    return G_pti

def getImagesFromDir(directory:str, device:torch.device, verbose:bool=True, dezired_size:int=-1):
    images = []
    for entry in os.scandir(directory):
        target_pil = PIL.Image.open(entry.path).convert('RGB')
        w, h = target_pil.size
        s = min(w, h)
        target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        if dezired_size<=0: dezired_size=s
        target_pil = target_pil.resize((dezired_size, dezired_size), PIL.Image.LANCZOS)
        img_np = np.array(target_pil, dtype=np.uint8)
        images.append(torch.tensor(img_np.transpose([2, 0, 1]), device=device))
    return images

def fitting(**kwargs):
    start_time = perf_counter()
    opts = dnnlib.EasyDict(kwargs)
    device = torch.device(opts.device)
    G = loadNetwork(opts.network_pkl, device, opts.verbose)
    dataset = ImagesDataset(
        opts.target_fname,
        None,
        G.img_resolution,
        opts.ips,
        opts.verbose
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    initPlugins(opts.verbose)
    os.makedirs(opts.outdir, exist_ok=True)
    """w_pivots = calculLatents(
        G,
        dataloader,
        device,
        opts.first_inv_steps,
        opts.inv_steps,
        opts.w_init,
        opts.save_latent,
        opts.save_video_latent,
        opts.outdir,
        opts.ips
    )
    G = pti_multiple_targets(
        G,
        w_pivots,
        dataloader,
        device,
        opts.pti_steps,
        verbose=opts.verbose,
        seed=opts.seed,
        save_video=opts.save_video,
        outdir=opts.outdir,
        disable_gradient_reg_loss=opts.disable_gradient_reg_loss
    )"""
    coache = MultiIDCoach(device, dataloader, opts.network_pkl, opts.outdir, opts.save_latent, opts.save_video_latent, opts.save_video, opts.seed)
    coache.train(opts.first_inv_steps, opts.inv_steps, opts.pti_steps)
    snapshot_data = {'G': G, 'G_ema': G}
    with open(f"{opts.outdir}/network.pkl", 'wb') as f:
        dill.dump(snapshot_data, f)
    if opts.verbose : print(f'Elapsed time: {(perf_counter()-start_time):.1f} s')
    

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--target', 'target_fname', help='Target video file or directory content target images', required=True, metavar='FILE')
@click.option('--seed', help='Random seed', type=int, default=64, show_default=True)
@click.option('--save-video', help='Save an mp4 video of fitting progress', type=bool, default=False, show_default=True, is_flag=True)
@click.option('--save-latent', help='Save latent in file npz', type=bool, default=False, show_default=True, is_flag=True)
@click.option('--save-video-latent', help='Save video for result latent optimization', type=bool, default=False, show_default=True, is_flag=True)
@click.option('--outdir', help='Where to save the output images', required=True, metavar='DIR', default='out')
@click.option('--first-inv-steps', help='Number of inversion steps for first image', type=int, default=1000, show_default=True)
@click.option('--inv-steps', help='Number of inversion steps for image (for first image use --first-inv-steps)', type=int, default=100, show_default=True)
@click.option('--w-init', help='path to inital latent', type=str, default='', show_default=True)
@click.option('--pti-steps', help='Number of pti steps', type=int, default=2500, show_default=True)
@click.option('--ips', help='The number of image used in one second of video', type=int, default=10, show_default=True)
@click.option('--not-verbose', 'verbose', help='this flag disable the verbose mode', default=True, is_flag=True)
@click.option('--device', help='torch device used', default='cuda', metavar='torch.device')
@click.option('--disable-gradient-reg-loss', help='disable gradient reg loss (not recommanded (quality of fitting is degraded), use only if you have out of memory)', default=False, is_flag=True)
def main(**kwargs):
    fitting(**kwargs)

if __name__ == "__main__":
    main()
