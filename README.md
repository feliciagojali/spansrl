# Span Based Semantic Role Labeling
A span-based Semantic Role Labeling model in bahasa Indonesia adapted from Li, et al. (2019) [[1]](#1) and built with keras tensorflow

# Installations
```
pip install -r requirements.txt
```

# Datasets
Datasets used to train SRL models in this repo can be fetched by running this script
```
make fetch_data
```
The fetched datas include srl data in particular format (raw) and its features to prepare for training.


# Testing with pretrained SRL models
1. Download pretrained models
All models can be downloaded from [here], please remember to put the pretrained models into `models` folder in root.
2. Download pretrained embedding models that will be used to extract features from sentences. The pretrained embedding models can be fetched by running this script.
```
make fetch_emb
```
3. Modify the `configurations.json` so that it fits with the pretrained model that is going to be used. Default config such as maximum tokens, spans, etc can be seen from `default` in `configurations.json`. The difference between pretrained models can be inferred from the models name such as BiHLSTM layers, unit, etc.
4. Run predict script. 
```
make predict config=$(config) file=$(filename)
```
There are two arguments that you can fill. First, config name, the default is `default` if you do not fill anything and the second one is output filename, the default filename will be `result.txt` and you can find the file in `data/results`. You can also choose between entering sentences one by one by command line or from file.

# Training with your own data
1. Make sure your data format follows the data format used in this module, example format can be seen in `data/raw/example.txt` (or you can create your own pre process code)
2. To get all features needed for training, modify the configuration.json to your own path etc.
3. Run the script to process your raw data to the features
```
make extract_features config=$(config)
```
4. This will results in all the features saved in your features directory (train, test, val with ratio 60:20:20)
5. Now you can start training the model with your features! Modify the hyperparameters in the config and make sure the features directory and filenames is correct. Run the training script.
```
make train config=$(config)
```
You can see the log while training in $(config).txt in root folder. The training script will also evaluate the model with your validation data, and you can see the scores in the log and the results in `data/results/$(model_name)`

6. If you want to test model with your validation data or test data, you can run either of the following scripts
```
make validate config=$(config)
```
or
```
make test config=$(config)
```

# References
<a id="1">[1]</a> 
Li, Z., He, S., Hai, Z., Zhang, Y., Zhang, Z., Zhou, X., & Zhou, X. (2019). Dependency or Span, End-to-End Uniform Semantic Role Labeling. ArXiv, abs/1901.05280.