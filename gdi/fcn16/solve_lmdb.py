import caffe
import sys

sys.path.append('..')
import surgery, score

import numpy as np
import os

weights = '../scp-voc-fcn32s-zoyaedits/fcn32_lr00001_trial2/_iter_100000.caffemodel'

# init
caffe.set_device(int(sys.argv[1]))
caffe.set_mode_gpu()

solver = caffe.SGDSolver('solver_lmdb.prototxt')
solver.net.copy_from(weights)

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)

solver.solve()