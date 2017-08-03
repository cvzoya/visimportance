import scipy.io
import numpy as np
from PIL import Image
import os 
import sys
import caffe 
import lmdb
import random
random.seed(0)

def load_image(maindir,idx,meanval,split):
    """
            Load input image and preprocess for Caffe:
            - cast to float
            - switch channels RGB -> BGR
            - subtract mean
            - transpose to channel x height x width order
            """
    
    if split=='train':
        im = Image.open('{}/GDI/gd_train/{}.jpg'.format(maindir, idx))
    else:
        im = Image.open('{}/GDI/gd_val/{}.jpg'.format(maindir, idx))

    in_ = np.array(im, dtype=np.float)
    #print('image:{}, size:{}'.format(idx,in_.shape))

    if len(in_.shape) < 3:
        w, h = in_.shape
        ret = np.empty((w, h, 3), dtype=np.float)
        ret[:, :, :] = in_[:, :, np.newaxis]
        in_ = ret
        # get rid of alpha dimension
        #im.load()
        
    if in_.shape[2] == 4:
        background = Image.new("RGB", im.size, (255, 255, 255))
        background.paste(im, mask=im.split()[3]) # 3 is the alpha channel
        in_ = np.array(background, dtype=np.float)

    in_ = in_[:,:,::-1]
    in_ -= meanval
    in_ = in_.transpose((2,0,1))
    return in_

        

def load_label(maindir, idx, split):
    """
    Load label image as 1 x height x width integer array of label indices.
    The leading singleton dimension is required by the loss.
    """

    if split=='train':
        im = Image.open('{}/GDI/gd_imp_train/{}.png'.format(maindir, idx)) 
    else:
        im = Image.open('{}/GDI/gd_imp_val/{}.png'.format(maindir, idx))
        
    label = np.array(im, dtype=np.uint8) 
    label = label/255.0
        
    label = label[np.newaxis, ...]
    return label
    
    
    
def DumpLMDB(OUT_DIR,maindir,split,randomize):
    
    meanval=(104.00699, 116.66877, 122.67892)
    
    if(not os.path.exists(OUT_DIR)):
        os.makedirs(OUT_DIR)

    # open lmdb dirs    
    fname_img_lmdb = os.path.join(OUT_DIR,'img.lmdb')
    fname_map_lmdb = os.path.join(OUT_DIR,'map.lmdb')
    
    img_env = lmdb.open(fname_img_lmdb, map_size=1e13)
    map_env = lmdb.open(fname_map_lmdb, map_size=1e13)
    
    # load indices for images and labels
    # make eval deterministic
    if split == 'train':
        split_f  = '{}/GDI/train.txt'.format(maindir)
        randomize = True
    else:
        split_f  = '{}/GDI/valid.txt'.format(maindir)
        randomize = False
    
    indices = open(split_f, 'r').read().splitlines()
    
    # randomization: seed and pick
    if randomize:
        #idx = random.randint(0, len(indices)-1)
        random.shuffle(indices) # random works in place and returns none

    with img_env.begin(write=True) as img_txn, map_env.begin(write=True) as map_txn:
        ii = 0
        for idx in indices:
            print 'Processing image '+ str(ii)
            key = '%06d' % (ii)
            
            img_cur = load_image(maindir,idx,meanval,split);
            map_cur = load_label(maindir,idx,split);
            
            img_datum = caffe.io.array_to_datum(img_cur)
            img_txn.put(key, img_datum.SerializeToString())
            
            map_datum = caffe.io.array_to_datum(map_cur)
            map_txn.put(key, map_datum.SerializeToString())
            
            ii += 1
    return
            

maindir = '../../data/'; # CHANGETHIS to where to save the LMDB database

# create validation LMDB
OUT_DIR = maindir+'GDI/valid_lmdb/';
split = 'valid';
randomize = False; 
DumpLMDB(OUT_DIR,maindir,split,randomize)

# create train set LMDB
OUT_DIR = maindir+'GDI/train_lmdb/'; 
split = 'train';
randomize = True; 
DumpLMDB(OUT_DIR,maindir,split,randomize)




