# pytorch-nlp
PyTorch Implementation for Natural Langauge Processing

<br/>

---
## Classification
- charcnn: Character-level Convolutional Networks for Text Classification [blog](https://www.notion.so/daangn/Character-level-Convolutional-Networks-for-Text-Classification-3fb5552c27b94a3099e8e79ba1a272f9)
- deepcnn: Very Deep Convolutional Networks for Text Classification [blog](https://www.notion.so/daangn/Very-Deep-Convolutional-Networks-for-Text-Classification-63c3f055d19b4a1285891c99f5b06517)
- lstmcnn: Efficient Character-level Document Classification by Combining Convolution and Recurrent Layers [blog](https://www.notion.so/daangn/Efficient-Character-level-Document-Classification-by-Combining-Convolution-and-Recurrent-Layers-a05e07dcbd0249978dd2d35504653577)

### How to train model
If you run code like this, then the Amazon Review dataset will be downloaded and the model you choose will be trained. It took about 1 day to train these models. 

```
cd classification
python train.py --name 'name of logs' --model 'with model to run' --gpu 'which gpu to use'
```

### Training result
- charcnn
  - number of parameters: 11,339,013
  - batch time: 0.251s (512 batch)
  - accuracy: 60.30 %
- deepcnn 
  - number of parameters: 16,444,005 개
  - batch time: 0.138s (128 batch)
  - accuracy: 62.85 %
- lstmcnn
  - number of parameters: 501381
  - batch time: 0.353s (512 batch)
  - accuracy: 59.61 %

<img src='https://user-images.githubusercontent.com/16641054/51795184-704f7a00-2222-11e9-97f5-c70d5f311f5d.png' width='600px'>

<br/>

---

## Language Model
- Character-Aware Neural Language Models: [paper](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/viewFile/12489/12017)

### How to train model
If you run code like this, then the Penn Treebank dataset will be downloaded and the model you choose will be trained. It took about 30 minute to train the models. 

```
cd language_model
python train.py --name 'name of logs' --gpu 'which gpu to use'
```

### How to test model
```
cd language_model
python test.py --model_path 'path to trained model path' --gpu 'which gpu to use'
```

### Training result
- character-aware neural language model
  - number of parameters: 5,312,485
  - batch time: 0.031 (20 batch)
  - perplexity on test dataset: 89.850 (paper: 92.3)
  
<img src='https://user-images.githubusercontent.com/16641054/52529241-ec24e880-2d31-11e9-93c4-cba52b618fd5.png' width='600px'>
