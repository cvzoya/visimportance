# SAVE PREDICTIONS from model

from PIL import Image
import numpy as np
import caffe
import os

def preprocess_image(im):
    # preprocess image same way as for network
    # load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
    mean=np.array((104.00699, 116.66877, 122.67892))
    
    in_ = np.array(im, dtype=np.float32)
    
    if len(in_.shape) < 3:
        w, h = in_.shape
        ret = np.empty((w, h, 3), dtype=np.float32)
        ret[:, :, :] = in_[:, :, np.newaxis]
        in_ = ret
    
    # get rid of alpha dimension
    if in_.shape[2] == 4:
        background = Image.new("RGB", im.size, (255, 255, 255))
        background.paste(im, mask=im.split()[3]) # 3 is the alpha channel
        in_ = np.array(background, dtype=np.float32)
    
    in_ = in_[:,:,::-1]
    in_ -= mean
    in_ = in_.transpose((2,0,1))
    return in_

def calc_pred_importance(im_loc,net):
    im = Image.open(im_loc)
    in_ = preprocess_image(im)
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_
    net.forward()
    return net.blobs['loss'].data[0]

# which images to compute predictions for
imdir = "../../data/GDI/gd_val/" # CHANGETHIS
allfiles = os.listdir(imdir)

# where to save the predictions
savedir = "../../data/GDI/vgg16_preds/" # CHANGETHIS
os.makedirs(savedir)

net = caffe.Net('fcn16/deploy.prototxt','../models/gdi_fcn16.caffemodel',caffe.TEST) # CHANGETHIS

for filename in allfiles:
    
    if not filename.endswith('.jpg'):
        continue
    
    im_loc = imdir+filename
    
    pred_imp = calc_pred_importance(im_loc,net)
    data = pred_imp[0,...]
    
    #Rescale to 0-255 and convert to uint8
    rescaled = (255.0 / data.max() * (data - data.min())).astype(np.uint8)
    im_new = Image.fromarray(rescaled)

    pre, ext = os.path.splitext(filename)
    new_loc = savedir+pre+'.png'
    im_new.save(new_loc,"PNG")
