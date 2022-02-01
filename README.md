# Span Based Semantic Role Labeling
A span-based Semantic Role Labeling model in bahasa Indonesia adapted from Li, et al. (2019) [[1]](#1) and built with keras tensorflow

# Installations

# Testing with pretrained models 

# Datasets and pretrained
All datasets and pretrained models can be fetched by running 
```
./src/data/fetch_all_datas.sh
```
The fetched datas include srl data in particular format, pretrained word embeddings (fasttext and word2vec)


# Training with your own data
1. Make sure your data format follows the data format used in this module (or you can create your own pre process code)
2. To get all features needed for training, modify the configuration.json to your own path etc.
3. Run the script to process your raw data to the features
```
python src/features/main.py
```
Please remember to modify the script as you needed as the scripts include reading srl data with its labels, process the sentences to features and also process the output to follow the model output,  modify if you only need to process the input and not the output

## References
<a id="1">[1]</a> 
Li, Z., He, S., Hai, Z., Zhang, Y., Zhang, Z., Zhou, X., & Zhou, X. (2019). Dependency or Span, End-to-End Uniform Semantic Role Labeling. ArXiv, abs/1901.05280.