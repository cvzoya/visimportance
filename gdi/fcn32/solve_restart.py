import caffe
import sys
import surgery
import numpy as np
import os

weights = 'snapshots/snapshot_fcn32_iter_24000.caffemodel' # CHANGETHIS

# init
caffe.set_device(int(sys.argv[1]))
caffe.set_mode_gpu()

solver = caffe.SGDSolver('solver.prototxt')
solver.restore('snapshots/snapshot_fcn32_iter_24000.solverstate') # CHANGETHIS
solver.net.copy_from(weights)

solver.solve()

