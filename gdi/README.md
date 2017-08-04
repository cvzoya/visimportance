# Training the GDI (Graphic Design Importance) Model

Similarly to the training procedure used for [semantic segmentation](https://github.com/shelhamer/fcn.berkeleyvision.org), we first trained an [FCN-32s model](https://github.com/cvzoya/visimportance/tree/master/gdi/fcn32) (using code in `fcn32`), and used it to initialize an [FCN-16s model](https://github.com/cvzoya/visimportance/tree/master/gdi/fcn16) with an extra skip connection to learn finer-grained structures (using code in `fcn16`). The FCN-32s model was initialized with the pre-trained [VOC-FCN32s](https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/voc-fcn32s/caffemodel-url). We used the custom data loaders specified in [imp_layers.py](https://github.com/cvzoya/visimportance/blob/master/gdi/imp_layers.py). 

We also provide code where data is loaded from an LMDB database in `fcn16_lmdb`, and all required data pre-processing steps occur during database construction: [create_lmdb_data.py](https://github.com/cvzoya/visimportance/blob/master/gdi/fcn16_lmdb/create_lmdb_data.py). This version of training does not require the custom data loaders in `imp_layers.py`. To use the code in `fcn16_lmdb`, make sure to first run `create_lmdb_data.py` in order to create the LMDB databases from the image datasets. 

To begin training, [download](https://github.com/cvzoya/visimportance/tree/master/data) the GDI dataset images and ground truth importance maps, update the paths to the data and pre-trained models, and run `solve.py`.

We also provide sample code for re-starting a terminated run: `solve_restart.py`. Update the file paths to use.

We provide code for computing the predictions using a trained model: [get_predictions.py](https://github.com/cvzoya/visimportance/blob/master/gdi/get_predictions.py). You can train your own or use one of our [pre-trained models](https://github.com/cvzoya/visimportance/tree/master/models).
