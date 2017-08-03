# if training crashes, can run this file

import caffe
import sys

sys.path.append('..')
import surgery, score

import numpy as np
import os

# init
caffe.set_device(int(sys.argv[1]))
caffe.set_mode_gpu()

solver = caffe.SGDSolver('solver_lmdb.prototxt')

solver.restore('snapshots/iter_30000.solverstate')

solver.solve()
