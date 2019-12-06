from fastai.torch_core import *
import librosa
from scipy.io import wavfile
from IPython.display import display, Audio

# For Kaggle as soundfile is not in Kaggle docker image
import importlib
soundfile_spec = importlib.util.find_spec("soundfile")
if soundfile_spec is not None:
    import soundfile as sf

__all__ = ['AudioClip', 'open_audio']

from .transform import MyDataAugmentation

class AudioClip(ItemBase):
    def __init__(self, signal, sample_rate, fn):
        self.data = signal # Contains original signal to start 
        self.original_signal = signal.clone()
        self.processed_signal = signal.clone()
        self.sample_rate = sample_rate
        self.fn = fn

    def __str__(self):
        return '(duration={}s, sample_rate={:.1f}KHz)'.format(
            self.duration, self.sample_rate/1000)

    def clone(self):
        return self.__class__(self.data.clone(), self.sample_rate, self.fn)

    def apply_tfms(self, tfms, **kwargs):
        for tfm in tfms:
            self.data = tfm(self.data, **kwargs)
            if issubclass(type(tfm), MyDataAugmentation):
                self.processed_signal = self.data.clone().cpu()
        return self
    
    @property
    def num_samples(self):
        return len(self.data)

    @property
    def duration(self):
        return self.num_samples / self.sample_rate

    def show(self, ax=None, figsize=(5, 1), player=True, title=None, **kwargs):
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        if title:
            ax.set_title("Class: " + str(title) + " \nfilename: " + str(self.fn))
        
        timesteps = np.arange(self.original_signal.shape[1]) / self.sample_rate
        
        ax.plot(timesteps, self.original_signal[0]) 
        if self.original_signal.size(0) > 1: # Check if mono or stereo
            ax.plot(timesteps, self.original_signal[1]) 
        ax.set_xlabel('Original Signal Time (s)')
        plt.show()
        
        timesteps = np.arange(self.processed_signal.shape[1]) / self.sample_rate

        _, ax = plt.subplots(figsize=figsize)
        if title:
            ax.set_title("Class: " + str(title) + " \nfilename: " + str(self.fn))
        ax.plot(timesteps, self.processed_signal[0]) 
        if self.processed_signal.size(0) > 1: # Check if mono or stereo
            ax.plot(timesteps, self.processed_signal[1]) 
        ax.set_xlabel('Processed Signal Time (s)')
        plt.show()
        
        if player:
            # unable to display an IPython 'Audio' player in plt axes
            display("Original signal")
            display(Audio(self.original_signal, rate=self.sample_rate))
            display("Processed signal")
            display(Audio(self.processed_signal, rate=self.sample_rate))

           

def open_audio(fn, using_librosa:bool=False, downsampling=8000):
    if using_librosa: 
        x, sr = librosa.core.load(fn, sr=None, mono=False)
        
    else:
        if soundfile_spec is not None:
            x, sr = sf.read(fn, always_2d=True, dtype="float32")
        else:
            raise Exception("Cannot load soundfile")
            #sr, x = wavfile.read(fn) # 10 times faster than librosa but issues with 24bits wave
    
    if len(x.shape) == 1: # Mono signal
        x = x.reshape(1, -1) # Add 1 channel
    else:
        if not using_librosa:
            x = np.swapaxes(x, 1, 0) # Scipy result is timestep * channels instead of channels * timestep
    
    if downsampling is not None:
        x = librosa.core.resample(x, sr, downsampling)
        sr = downsampling
    t = torch.from_numpy(x.astype(np.float32, copy=False))
    if x.dtype == np.int16:
        t.div_(32767)
    elif x.dtype != np.float32:
        raise OSError('Encountered unexpected dtype: {}'.format(x.dtype))
    return AudioClip(t, sr, fn)
