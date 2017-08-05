# Plotting the training and validation curves

When training the models, caffe will generate output of the training and validation error over time. If you pipe this output to a log file, then you can use the scripts provided here to parse this training output and plot it.

For example: `python solve.py 1 2>&1 | tee training_log.txt` will pipe the training output to `training_log.txt`

Then, `parselog2.sh` can be used to parse this log, and the functions in `get_learning_curves.py` to plot simple training and validation curves. Feel free to use this as a starter code for visualizing the progress of your training.

<img src="training_error.png" height="200">
