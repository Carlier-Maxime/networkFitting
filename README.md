# NetworkFitting

NetworkFitting is used to fit a StyleGAN type network for more specific images than the dataset it was trained on, for example if the network was trained to generate faces using the FFHQ dataset. Then you can use networkFitting to make the network fit more precisely for a given person by using several images of the person or a video of it.

This program was designed to fit a StyleGAN network that generates faces with tags on a video of a particular person with tags to obtain a network that can be reused to automatically generate tags on a video of the person without markers and thus obtain this same video with markers