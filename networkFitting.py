import click, torch
from importlib import __import__ as import_

def fromImport(module:str, name:str):
    return getattr(import_(module, fromlist=[name]),name)

def importAll():
    global dnnlib, legacy
    dnnlib = fromImport('stylegan-xl','dnnlib')
    legacy = fromImport('stylegan-xl','legacy')

def loadNetwork(network_pkl:str, device:torch.device, verbose:bool=True):
    if verbose: print('Loading networks from "%s"...' % network_pkl)
    device = torch.device(device)
    with dnnlib.util.open_url(network_pkl) as fp:
        G = legacy.load_network_pkl(fp)['G_ema'].to(device) # type: ignore

def fitting(**kwargs):
    importAll() #import module with bad name
    opts = dnnlib.EasyDict(kwargs) #get options (arguments given)
    loadNetwork(opts.network_pkl, opts.device, opts.verbose)

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--target', 'target_fname', help='Target video file to project to', required=True, metavar='FILE')
@click.option('--seed', help='Random seed', type=int, default=64, show_default=True)
@click.option('--save-video', help='Save an mp4 video of fitting progress', type=bool, default=False, show_default=True, is_flag=True)
@click.option('--save-latent', help='Save latent in file npz', type=bool, default=False, show_default=True, is_flag=True)
@click.option('--save-video-latent', help='Save video for result latent optimization', type=bool, default=False, show_default=True, is_flag=True)
@click.option('--outdir', help='Where to save the output images', required=True, metavar='DIR', default='out')
@click.option('--first-inv-steps', help='Number of inversion steps for first image', type=int, default=1000, show_default=True)
@click.option('--inv-steps', help='Number of inversion steps for image (for first image use --first-inv-steps)', type=int, default=100, show_default=True)
@click.option('--w-init', help='path to inital latent', type=str, default='', show_default=True)
@click.option('--pti-steps', help='Number of pti steps', type=int, default=2500, show_default=True)
@click.option('--v-fps', help='The number of frame used in one second of video', type=int, default=10, show_default=True)
@click.option('--not-verbose', 'verbose', help='this flag disable the verbose mode', default=True, is_flag=True)
@click.option('--device', help='torch device used', default='cuda', metavar='torch.device')
def main(**kwargs):
    fitting(**kwargs)

if __name__ == "__main__":
    main()
