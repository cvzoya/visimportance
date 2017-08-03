import caffe
import sys
import surgery
import numpy as np
import os

weights = 'fcn32s-heavy-pascal.caffemodel' # CHANGETHIS to location of pre-trained VOC FCN-32s model

# init
caffe.set_device(int(sys.argv[1]))
caffe.set_mode_gpu()

solver = caffe.SGDSolver('solver.prototxt')
solver.net.copy_from(weights)

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)

solver.solve()
