#!/usr/bin/env python
# coding: utf-8

# # PASCAL VOC
# 


# In[1]:



# In[2]:


import glob
#import sys
#sys.path.append('../../src/')
import os
from functools import partial


# In[3]:


from PIL import Image

import cv2
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


# In[4]:


from nn.encoders import mbv2
from nn.micro_decoders import MicroDecoder as Decoder
from utils.helpers import prepare_img
from utils.model_zoo import load_url


# In[5]:
class ScrollableWindow(QtWidgets.QMainWindow):
    def __init__(self, fig):
        self.qapp = QtWidgets.QApplication([])

        QtWidgets.QMainWindow.__init__(self)
        self.widget = QtWidgets.QWidget()
        self.setCentralWidget(self.widget)
        self.widget.setLayout(QtWidgets.QVBoxLayout())
        self.widget.layout().setContentsMargins(0,0,0,0)
        self.widget.layout().setSpacing(0)

        self.fig = fig
        self.canvas = FigureCanvas(self.fig)
        self.canvas.draw()
        self.scroll = QtWidgets.QScrollArea(self.widget)
        self.scroll.setWidget(self.canvas)

        self.nav = NavigationToolbar(self.canvas, self.widget)
        self.widget.layout().addWidget(self.nav)
        self.widget.layout().addWidget(self.scroll)

        self.show()
        exit(self.qapp.exec_()) 

class Segmenter(nn.Module):
    """Create Segmenter"""
    def __init__(self, encoder, decoder):
        super(Segmenter, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def _reset_clf(self, num_classes):
        self.decoder._reset_clf(num_classes)

    def forward(self, x):
        return self.decoder(self.encoder(x))


# In[6]:


models_info = {
    'arch0_segm-23378522.pth' : [
        'https://cloudstor.aarnet.edu.au/plus/s/ZPXVGGgyxekvdAn/download',
        [[8, [0, 0, 5, 2], [0, 2, 8, 8], [0, 5, 1, 4]], [[3, 3], [3, 2], [3, 0]]]
    ],
    'arch1_segm-12f78b21.pth' : [
        'https://cloudstor.aarnet.edu.au/plus/s/svINhJX7IsvjCaD/download',
        [[2, [1, 0, 3, 6], [0, 1, 2, 8], [2, 0, 6, 1]], [[2, 3], [3, 1], [4, 4]]]
    ],
    'arch2_segm-8f00fc4d.pth' : [
        'https://cloudstor.aarnet.edu.au/plus/s/9b8zVuaowe6ZtAN/download',
        [[5, [0, 0, 4, 1], [3, 2, 0, 1], [5, 6, 5, 0]], [[1, 3], [4, 3], [2, 2]]]
    ]
}


# In[7]:


# Configuration
AGG_SIZE = 64
AUX_CELL = False
REPEATS = 2
NUM_CLASSES = 21


# In[8]:


cmap = np.load('./utils/cmap.npy')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dec_fn = partial(Decoder,
                 num_classes=NUM_CLASSES,
                 agg_size=AGG_SIZE,
                 aux_cell=AUX_CELL,
                 repeats=REPEATS)
img_dir = './imgs/VOC/'
imgs = glob.glob('{}*.jpg'.format(img_dir))


# In[9]:


# Initialise models
models = dict()

for name, urlconfig in models_info.items():
    arch = name.split('-')[0]
    url, config = urlconfig
    
    enc = mbv2(pretrained=False)
    dec = dec_fn(config=config, inp_sizes=enc.out_sizes)
    segm = Segmenter(enc, dec).to(device).eval()
    modelpath = os.path.join(".","models",name)
    segm.load_state_dict(torch.load(modelpath, map_location=device), strict=False)
    models[arch] = segm
    del enc


# In[10]:


# Figure 7 from the paper
n_cols = len(models) + 2 # 1 - for image, 1 - for GT
n_rows = len(imgs)


#fig = plt.figure()
fig,ax = plt.subplots(n_rows, n_cols, figsize=(3*n_cols,3*n_rows))
fig.set_layout_engine(layout='tight')
idx = 0
with torch.no_grad():
    for img_path in imgs:
        img = np.array(Image.open(img_path))
        msk = cmap[np.array(Image.open(img_path.replace('jpg', 'png')))]
        orig_size = img.shape[:2][::-1]
        
        img_inp = torch.tensor(
            prepare_img(img).transpose(2, 0, 1)[None]).float().to(device)
        
        
        ax[idx,0].imshow(img)
        ax[idx,0].set_title('data')
        ax[idx,0].axis('off')
        
        
        
        ax[idx,1].imshow(msk)
        ax[idx,1].set_title('best')
        ax[idx,1].axis('off')
        k=2
        for mname, mnet in models.items():
            segm = mnet(img_inp)[0].squeeze().data.cpu().numpy().transpose((1, 2, 0))
            segm = cv2.resize(segm, orig_size, interpolation=cv2.INTER_CUBIC)
            segm = cmap[segm.argmax(axis=2).astype(np.uint8)]
            
            
            ax[idx,k].imshow(segm)
            ax[idx,k].set_title("model "+mname)
            ax[idx,k].axis('off')
            k += 1
        idx += 1

scrollPlot = ScrollableWindow(fig);
# In[ ]:




