import sys
sys.path.append('../../caffe/python')

import caffe
from caffe import layers as L, params as P
from caffe.coord_map import crop

# set base net learning rate (for instance if want to freeze it):
frozen_param = [dict(lr_mult=0)] * 2
weight_param = dict(lr_mult=1, decay_mult=1)
bias_param   = dict(lr_mult=2, decay_mult=0)
learned_param = [weight_param, bias_param]
tobinarize = 0;
# --------------------------

def conv_relu(bottom, nout, ks=3, stride=1, pad=1, param=learned_param):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
        num_output=nout, pad=pad, param=param)
    return conv, L.ReLU(conv, in_place=True)

def max_pool(bottom, ks=2, stride=2):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def fcn(split,learn_all=False):
    
    param = learned_param if learn_all else frozen_param
    
    n = caffe.NetSpec()
    pydata_params = dict(split=split, mean=(104.00699, 116.66877, 122.67892),
            seed=1337)
    if split == 'train':
        pydata_params['train_dir'] = '../../data/'
        pylayer = 'GDIDataLayerImp' # can instead choose: MassvisDataLayerBubble
        pydata_params['binarize'] = tobinarize
    else:
        pydata_params['val_dir'] = '../../data/'
        pylayer = 'GDIValDataLayer'
        pydata_params['binarize'] = tobinarize
        
        
    n.data, n.label = L.Python(module='imp_layers', layer=pylayer,
            ntop=2, param_str=str(pydata_params))

    # the base net
    n.conv1_1, n.relu1_1 = conv_relu(n.data, 64, pad=100, param=param)
    n.conv1_2, n.relu1_2 = conv_relu(n.relu1_1, 64, param=param)
    n.pool1 = max_pool(n.relu1_2)

    n.conv2_1, n.relu2_1 = conv_relu(n.pool1, 128, param=param)
    n.conv2_2, n.relu2_2 = conv_relu(n.relu2_1, 128, param=param)
    n.pool2 = max_pool(n.relu2_2)

    n.conv3_1, n.relu3_1 = conv_relu(n.pool2, 256, param=param)
    n.conv3_2, n.relu3_2 = conv_relu(n.relu3_1, 256, param=param)
    n.conv3_3, n.relu3_3 = conv_relu(n.relu3_2, 256, param=param)
    n.pool3 = max_pool(n.relu3_3)

    n.conv4_1, n.relu4_1 = conv_relu(n.pool3, 512, param=param)
    n.conv4_2, n.relu4_2 = conv_relu(n.relu4_1, 512, param=param)
    n.conv4_3, n.relu4_3 = conv_relu(n.relu4_2, 512, param=param)
    n.pool4 = max_pool(n.relu4_3)

    n.conv5_1, n.relu5_1 = conv_relu(n.pool4, 512, param=param)
    n.conv5_2, n.relu5_2 = conv_relu(n.relu5_1, 512, param=param)
    n.conv5_3, n.relu5_3 = conv_relu(n.relu5_2, 512, param=param)
    n.pool5 = max_pool(n.relu5_3)

    # fully conv
    n.fc6, n.relu6 = conv_relu(n.pool5, 4096, ks=7, pad=0, param=param)
    n.drop6 = L.Dropout(n.relu6, dropout_ratio=0.5, in_place=True)
    n.fc7, n.relu7 = conv_relu(n.drop6, 4096, ks=1, pad=0, param=param)
    n.drop7 = L.Dropout(n.relu7, dropout_ratio=0.5, in_place=True)
    
    # the following are parameters being learnt anyway
    # made layer named score_sal, upscore_sal to avoid copying new weights
    n.score_sal = L.Convolution(n.drop7, num_output=1, kernel_size=1, pad=0,
                                param=learned_param) # <- learning weights for this layer

    # CHANGES DUE TO SKIP CONNECTION:
    # don't upscale all the way, only to the previous layer
    # replaced kernel_size=64, stride=32 with:
    n.upscore_sal2 = L.Deconvolution(n.score_sal,
        convolution_param=dict(num_output=1, kernel_size=4, stride=2,
            bias_term=False),param=[dict(lr_mult=0)]) # don't learn upscoring; fix it as bilinear

    n.score_pool4 = L.Convolution(n.pool4, num_output=1, kernel_size=1, pad=0, param=learned_param) # <- learning weights for this layer
    n.score_pool4c = crop(n.score_pool4, n.upscore_sal2)
    n.fuse_pool4 = L.Eltwise(n.upscore_sal2, n.score_pool4c, operation=P.Eltwise.SUM)
    n.upscore16 = L.Deconvolution(n.fuse_pool4,
        convolution_param=dict(num_output=1, kernel_size=32, stride=16,
            bias_term=False),param=[dict(lr_mult=0)])
    # don't learn any of the upscaling (deconvolution) filters - just fix at bilinear
                                  

    # n.score = crop(n.upscore_sal, n.data)
    n.score = crop(n.upscore16, n.data)

    n.loss = L.SigmoidCrossEntropyLoss(n.score, n.label, loss_weight=1) 

    return n.to_proto()

def make_net():
    
    learn_all=False
    
    with open('train.prototxt', 'w') as f:
        f.write(str(fcn('train',learn_all)))

    with open('val.prototxt', 'w') as f:
        f.write(str(fcn('valid',learn_all)))

if __name__ == '__main__':
    make_net()
