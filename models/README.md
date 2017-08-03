# Downloading models

## GDI (Graphic Design Importance) Dataset

  * FCN-32s: [gdi_vgg32.caffemodel](http://visimportance.mit.edu/data/GDI/gdi_vgg32.caffemodel)
  * FCN-16s (final model): [gdi_vgg16.caffemodel](http://visimportance.mit.edu/data/GDI/gdi_vgg16.caffemodel)

The FCN-16s model was initialized from the FCN-32s model. As specified in [fcn16/net.py](https://github.com/cvzoya/visimportance/blob/master/gdi/fcn16/net.py) it has an extra skip connection that was fine-tuned (all layers except score_sal and score_pool4 were frozen).

