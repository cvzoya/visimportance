# Training the Massvis Importance Model

We based our model on the [model for semantic segmentation](https://github.com/shelhamer/fcn.berkeleyvision.org), and trained an [FCN-32s model](https://github.com/cvzoya/visimportance/tree/master/models) by initializing it with the pre-trained [VOC-FCN32s](https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/voc-fcn32s/caffemodel-url). We used the custom data loaders specified in [imp_layers_massvis.py](https://github.com/cvzoya/visimportance/blob/master/massvis/imp_layers_massvis.py). 

To begin training, [download the Massvis dataset images](https://github.com/cvzoya/visimportance/tree/master/data) and ground truth importance maps, update the paths to the data and pre-trained models, and start training: `python solve.py N` (where N is replaced by the desired GPU ID).

We also provide code for computing the predictions using a trained model: [get_predictions.py](https://github.com/cvzoya/visimportance/blob/master/massvis/get_predictions.py). You can train your own or download `massvis_fcn32.caffemodel` from our pre-trained [models](https://github.com/cvzoya/visimportance/tree/master/models).
