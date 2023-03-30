import click
from importlib import __import__ as import_

def importAll():
    global dnnlib
    dnnlib = getattr(import_('stylegan-xl',fromlist=['dnnlib']),'dnnlib')

def fitting(**kwargs):
    importAll() #import module with bad name
    opts = dnnlib.EasyDict(kwargs) #get options (arguments given)
    print(opts.network_pkl, opts.target_fname, opts.save_video, opts.outdir)
    print('fitting..')

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
def main(**kwargs):
    fitting(**kwargs)

if __name__ == "__main__":
    main()
