# Downloading models

## GDI (Graphic Design Importance) Dataset

  * FCN-32s: [gdi_fcn32.caffemodel](http://visimportance.mit.edu/data/GDI/gdi_fcn32.caffemodel)
  * FCN-16s (final model): [gdi_fcn16.caffemodel](http://visimportance.mit.edu/data/GDI/gdi_fcn16.caffemodel)

The FCN-16s model was initialized from the FCN-32s model. As specified in [fcn16/net.py](https://github.com/cvzoya/visimportance/blob/master/gdi/fcn16/net.py) it has an extra skip connection that was fine-tuned (all layers except score_sal and score_pool4 were frozen).

## Massvis (Visualization) Dataset

  * FCN-32s: [massvis_fcn32.caffemodel](http://visimportance.mit.edu/data/massvis/massvis_fcn32.caffemodel)
