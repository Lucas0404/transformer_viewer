# Transformer Viewer
Simple visualization for pytorch model, Test version for classification task  
Implementation of paper:   [Axiomatic Attribution for Deep Networks](https://arxiv.org/abs/1703.01365)

## Requirments:
* python > 3.6  
* pytorch > 1.4  
* Colr > 0.9  

## Installation:
    pip install transformer_viewer

## How to use:
```python
import torch
import json

from transformer_viewer import Glimpse


# Load dict
with open(PATH_ID2LABEL) as infile: id2label = json.load(infile)
with open(PATH_ID2WORD) as infile: id2word = json.load(infile)
word2id = dict()
for k, v in id2word.items():
    word2id[int(v)] = k

# Load model
model = MyModel(*args, **kwargs)
model.load_state_dict(torch.load(PATH_MODEL))

tokenizer = lambda text: [int(word2id[item]) for item in text.split(" ")]
special_tokens = [1, 2] # 1 for <eos>, 2 for <pad>

viewer = Glimpse(model, "embeddings", id2word, id2label, tokenizer, special_tokens, loss_pos=0)

viewer.color_bar()
```
![color bar](./img/bar.png)

```python
viewer.view("乌鲁木齐市 新增 一处 城市 中心 旅游 目的地", "travel")
```
![true example](./img/true.png)
```python
viewer.view("郭晶晶 曾 撮合 吴敏霞 与 章子怡 前男友 ， 拒绝 豪门 平淡 才 是 真", "sports")
```
![wrong example](./img/wrong.png)

## Parameters:
### Glimplse(model, embed_name, id2word, id2label, tokenizer, special_tokens, loss_pos=None, step=20)
parameter|type|description|example
---|:--:|:--:|---:
model|object|pytorch model|
embed_name|str|name of the embedding layer|'embeddings'
id2word|dict|from id to token|{1: '你好'， 2, '再见'}
id2label|dict|from id to label|{1: 'sports'， 2, 'travel'}
tokenizer|function|which can convert a text to a index list|split
special_tokens|list|ids of the specical tokens|[1, 2]
loss_pos|int|position of loss for the output of model|0

### view(text, label)
parameter|type|description|example
---|:--:|:--:|---:
text|str|input text|'我 爱 中国'
label|int or str||'car'
