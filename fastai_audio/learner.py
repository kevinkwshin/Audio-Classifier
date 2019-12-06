from fastai.torch_core import *
from fastai.train import Learner

from fastai.callbacks.hooks import num_features_model, hook_output
from fastai.vision import create_body, create_head, Image
from fastai.vision.learner import cnn_config, _resnet_split, ClassificationInterpretation

__all__ = ['create_cnn']


# copied from fastai.vision.learner, omitting unused args,
# and adding channel summing of first convolutional layer
def create_cnn(data, arch, pretrained=False, is_mono_input=True, **kwargs):
    meta = cnn_config(arch)
    body = create_body(arch, pretrained)

    # sum up the weights of in_channels axis, to reduce to single input channel
    # Suggestion by David Gutman
    # https://forums.fast.ai/t/black-and-white-images-on-vgg16/2479/2
    if is_mono_input:
        first_conv_layer = body[0]
        first_conv_weights = first_conv_layer.state_dict()['weight']
        assert first_conv_weights.size(1) == 3 # RGB channels dim
        summed_weights = torch.sum(first_conv_weights, dim=1, keepdim=True)
        first_conv_layer.weight.data = summed_weights
        first_conv_layer.in_channels = 1
    else:
        # In this case, the input is a stereo
        first_conv_layer = body[0]
        first_conv_weights = first_conv_layer.state_dict()['weight']
        assert first_conv_weights.size(1) == 3 # RGB channels dim
        summed_weights = torch.sum(first_conv_weights, dim=1, keepdim=True)
        first_conv_layer.weight.data = first_conv_weights[:, :2, :, :] # Keep only 2 channels for the weights
        first_conv_layer.in_channels = 2

    nf = num_features_model(body) * 2
    head = create_head(nf, data.c, None, 0.5)
    model = nn.Sequential(body, head)
    learn = Learner(data, model, **kwargs)
    learn.split(meta['split'])
    if pretrained:
        learn.freeze()
    apply_init(model[1], nn.init.kaiming_normal_)
    return learn



def my_cl_int_plot_top_losses(self, k, largest=True, figsize=(25,7), heatmap:bool=True, heatmap_thresh:int=16,
                            return_fig:bool=None)->Optional[plt.Figure]:
    "Show images in `top_losses` along with their prediction, actual, loss, and probability of actual class."
    tl_val,tl_idx = self.top_losses(k, largest)
    classes = self.data.classes
    cols = math.ceil(math.sqrt(k))
    rows = math.ceil(k/cols)
    fig,axes = plt.subplots(rows, cols, figsize=figsize)
    fig.suptitle('prediction/actual/loss/probability', weight='bold', size=14)
    for i,idx in enumerate(tl_idx):
        audio, cl = self.data.dl(self.ds_type).dataset[idx]
        audio = audio.clone()
        
        m = self.learn.model.eval()
        
        x, _ = self.data.one_item(audio) # Process one audio into prediction
        
        x_consolidated = x.sum(dim=1, keepdim=True) # Sum accross all channels to ease the interpretation

        im = Image(x_consolidated[0, :, :, :].cpu()) # Extract the processed image from the prediction (after dl_tfms) and keep it into CPU
        cl = int(cl)
        title = f'{classes[self.pred_class[idx]]}/{classes[cl]} / {self.losses[idx]:.2f} / {self.probs[idx][cl]:.2f}'
        title = title + f'\n {audio.fn}'
        
        im.show(ax=axes.flat[i], title=title)
        
        if heatmap:
            # Related paper http://openaccess.thecvf.com/content_ICCV_2017/papers/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.pdf
            with hook_output(m[0]) as hook_a: # hook activations from CNN module
                with hook_output(m[0], grad= True) as hook_g: # hook gradients from CNN module
                    preds = m(x) # Forward pass to get activations
                    preds[0,cl].backward() # Backward pass to get gradients
            acts = hook_a.stored[0].cpu()
            if (acts.shape[-1]*acts.shape[-2]) >= heatmap_thresh:
                grad = hook_g.stored[0][0].cpu() # Hook the gradients from the CNN module and extract the first one (because one item only)
                grad_chan = grad.mean(1).mean(1) # Mean accross image to keep mean gradients per channel 
                mult = F.relu(((acts*grad_chan[...,None,None])).sum(0)) # Multiply activation with gradients (add 1 dim for height and width)
                sz = list(im.shape[-2:])
                axes.flat[i].imshow(mult, alpha=0.35, extent=(0,*sz[::-1],0), interpolation='bilinear', cmap='magma')     
        
    if ifnone(return_fig, defaults.return_fig): return fig
    
    
ClassificationInterpretation.plot_audio_top_losses = my_cl_int_plot_top_losses