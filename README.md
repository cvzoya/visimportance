# visimportance

Code to predict importance (saliency) on graphic designs and data visualizations.
We also provide links to our [models](https://github.com/cvzoya/visimportance/tree/master/models) and our train/test [data](https://github.com/cvzoya/visimportance/tree/master/data).

About our models:
  * We provide pre-trained models for both graphic design and data visualization importance prediction. These models were separately trained on the GDI and Massvis datasets, respectively. 
  * We initialized our models using the pre-trained [VOC-FCN32s](https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/voc-fcn32s/caffemodel-url) and only fine-tuned the final importance prediction layer.

About our data loaders:
  * We wrote custom data loaders in [imp_layers.py](https://github.com/cvzoya/visimportance/blob/master/imp_layers.py) which get invoked by the data layers (see top of train.prototxt and val.prototxt files)
  * We also provide an example of how to load data using a pre-constructed LMDB database, without relying on these custom data loaders (see [gdi/fcn16_lmdb](https://github.com/cvzoya/visimportance/tree/master/gdi/fcn16_lmdb)). In this case, all the data processing occurs during database construction (see [create_lmdb_data.py](https://github.com/cvzoya/visimportance/blob/master/gdi/fcn16_lmdb/create_lmdb_data.py))

If you use this code, please cite:
Zoya Bylinskii, Nam Wook Kim, Peter O'Donovan, Sami Alsheikh, Spandan Madan, Hanspeter Pfister, Fredo Durand, Bryan Russell, and Aaron Hertzmann. "Learning Visual Importance for Graphic Designs and Data Visualizations" (UIST'17)

