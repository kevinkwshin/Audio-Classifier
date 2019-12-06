from fastai.basic_data import *
from fastai.data_block import *
from fastai.data_block import _maybe_squeeze
from fastai.text import SortSampler, SortishSampler
from fastai.vision import *
from fastai.torch_core import *
from .audio_clip import *
import gc
import copy

__all__ = ['AudioDataBunch', 'AudioItemList', ]



class AudioDataBunch(DataBunch):
    
    # Subclass because of bug to give dl_tfms to underlying dataloader
    @classmethod
    def create(cls, train_ds, valid_ds, 
               tfms:Optional[Collection[Callable]]=None, # There is a bug in LabelLists because dl_tfms is not given to dataloader
               **kwargs)->'AudioDataBunch':
        db = super().create(train_ds=train_ds, valid_ds=valid_ds, dl_tfms=tfms, **kwargs)

        return db



    def show_batch(self, rows:int=5, ds_type:DatasetType=DatasetType.Train, **kwargs):
        dl = self.dl(ds_type)
        ds = dl.dl.dataset

        idx = np.random.choice(len(ds), size=rows, replace=False)
        batch = ds[idx]
        
        max_count = min(rows, len(batch))
        xs, ys, xs_processed, ys_processed = [], [], [], []
        for i in range(max_count):
            x, x_processed, y, y_processed = batch[i][0], batch[i][0].data, batch[i][1], torch.tensor(batch[i][1].data)
            xs.append(x)
            xs_processed.append(x_processed)
            ys.append(y)
            ys_processed.append(y_processed)

        xs_processed = torch.stack(xs_processed, dim=0)
        ys_processed = torch.stack(ys_processed, dim=0)
        
        for tfm in dl.tfms:
            xs_processed, ys_processed = tfm((xs_processed, ys_processed))

        
        self.train_ds.show_xys(xs, ys, xs_processed=xs_processed.unbind(dim=0), **kwargs)
        del xs, ys, xs_processed, ys_processed

    # Inspired by ImageDataBunch
    def batch_stats(self, funcs:Collection[Callable]=None, ds_type:DatasetType=DatasetType.Train)->Tensor:
        "Grab a batch of data and call reduction function `func` per channel"
        funcs = ifnone(funcs, [torch.mean,torch.std])
        x = self.one_batch(ds_type=ds_type, denorm=False)[0].cpu()
        return [func(channel_view(x), 1) for func in funcs]
        
    # Inspired by ImageDataBunch
    def normalize(self, stats:Collection[Tensor]=None, do_x:bool=True, do_y:bool=False)->None:
        "Add normalize transform using `stats` (defaults to `DataBunch.batch_stats`)"
        if getattr(self,'norm',False): raise Exception('Can not call normalize twice')
        if stats is None: self.stats = self.batch_stats()
        else:             self.stats = stats
        self.norm,self.denorm = normalize_funcs(*self.stats, do_x=do_x, do_y=do_y)
        self.add_tfm(self.norm)
        return self

       

# Inspired by https://docs.fast.ai/tutorial.itemlist.html
class AudioItemList(ItemList):
    _bunch = AudioDataBunch # Needed to include normalize
    
    def __init__(self, items:Iterator,
                 using_librosa=False, downsampling=None, **kwargs):
        super().__init__(items=items, **kwargs)
        self.using_librosa = using_librosa
        self.copy_new.append('using_librosa')
        self.downsampling = downsampling
        self.copy_new.append('downsampling')

    def get(self, i):
        fn = super().get(i)
        return open_audio(self.path/fn, using_librosa=self.using_librosa, downsampling=self.downsampling)


    @classmethod
    def from_df(cls, df, path, using_librosa=False, downsampling=None, **kwargs):
        res = super().from_df(df, path=path, **kwargs)
        res.using_librosa=using_librosa
        res.downsampling = downsampling
        return res
    
    
    def reconstruct(self, t:Tensor, x:Tensor = None): 
        raise Exception("Not implemented yet")
        # From torch
        #return ImagePoints(FlowField(x.size, t), scale=False)

    
    
    def show_xys(self, xs, ys, xs_processed=None, figsize=None, **kwargs):
        if xs_processed is None:
            for x, y in zip(xs, ys):
                x.show(title=str(y), figsize=figsize, **kwargs)
        else:
            for x, y, x_processed in zip(xs, ys, xs_processed):
                x.show(title=str(y), figsize=figsize, **kwargs)
                for channel in range(x_processed.size(0)):
                    Image(x_processed[channel, :, :].unsqueeze(0)).show(figsize=figsize)
