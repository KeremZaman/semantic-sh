# semantic-sh


semantic-sh is a SimHash implementation to detect and group similar texts by taking power of word vectors and transformer-based language models (BERT).



# Requirements
 - fasttext
 - transformers
 - pytorch
 - numpy
 - flask
# Installing via pip

```sh
$ pip install semantic-sh
```
# Notes

  - Only fasttext and huggingface BERT models are supported for now.
  - No batch processing support for now.


# Usage

```
from semantic_sh import SemanticSimHash
```

### Use with fasttext:
```
sh = SemanticSimHash(model_type='fasttext', dim=300, model_path='pat_to_fasttext_vectors.bin')
```

### Use with BERT:

```
sh = SemanticSimHash(model_type='bert-base-multilingual-cased', dim=768)
```

### Additional parameters

Customize threshold (default:0) , hash length (default: 256-bit) and add stop words list.

```
sh = SemanticSimHash(model_type='fasttext', key_size=128, dim=300, model_path='pat_to_fasttext_vectors.bin', thresh=0.8, stop_words=['the', 'i', 'you', 'he', 'she', 'it', 'we', 'they'])
```

**Note:** BERT-based models do not require stop words list.

### Hash your text

```
sh.get_hash('<your_text>')
```

### Add document

Add your document to the proper group

```
sh.add_document('<your_text>')
```

###  Find similar

Get all documents in the same group with the given text

```
sh.find_similar('<your_text>')
```

### Get Hamming Distance between 2 texts

```
sh.get_distance('<first_text>', '<second_text>')
```

### Go through all document groups

Get all similar document groups which have more than 1 document

```
for docs in sh.get_similar_groups():
   print(docs)
```

### Save data

Save added documents, hash function, model and parameters

```
sh.save('model.dat')
```

### Load from saved file

Load all parameters, documents, hash function and model from saved file

```
sh = SemanticSimHash.load('model.dat')
```

# API Server
Easily deploy a simple text similarity engine on web.

## Installation
```sh 
$ git clone https://github.com/KeremZaman/semantic-sh.git
```

## Standalone Usage
```
server.py [-h] [--host HOST] [--port PORT] [--model-type MODEL_TYPE]
                 [--model-path MODEL_PATH] [--key-size KEY_SIZE] [--dim DIM]
                 [--stop-words [STOP_WORDS [STOP_WORDS ...]]]
                 [--load-from LOAD_FROM]

optional arguments:
  -h, --help            show this help message and exit

app:
  --host HOST
  --port PORT

model:
  --model-type MODEL_TYPE
                        Type of model to run: fasttext or any pretrained model
                        name from huggingface/transformers
  --model-path MODEL_PATH
                        Path to vector files of fasttext models
  --key-size KEY_SIZE   Hash length in bits
  --dim DIM             Dimension of text representations according to chosen
                        model type
  --stop-words [STOP_WORDS [STOP_WORDS ...]]
                        List of stop words to exclude

loader:
  --load-from LOAD_FROM
                        Load previously saved state

```

## Using with WSGI Container
```
from gevent.pywsgi import WSGIServer
from server import init_app

app = init_app(params) # same params as initialize SemantcSimHash object

http_server = WSGIServer(('', 5000), app)
http_server.serve_forever()
```

**NOTE:** Sample code uses gevent but you can use any WSGI container which can be used with Flask app object instead.

## API Reference


```GET /api/hash?text=<text>```

Return hash of given text 

***

```GET /api/add?text=<text>```

Add given text as document
***

```GET /api/find-similar?text=<text>```

Return similar documents to given text
*** 

```GET /api/distance?src=<src_text>&tgt=<tgt_txt>```

Return Hamming distance between source and target texts
***

```GET /api/similarity-groups```

Return buckets having more than one document

## With docker

Run the api server on port 4000
```sh
docker run -ti -p 4000:4000 -v `pwd`/data:/opt/data  semantic-sh:latest --port=4000 --model-type=bert-base-multilingual-cased --model-path=/opt/data
```

## With docker-compose 
Run the api server on port 4000
```sh
docker-compose up -d semantic-sh
```

# Some Implementation Details

This is a simplified implementation of simhash by just creating random 
vectors and assigning 1 or 0 according to the result of dot product of each of these vectors with
represantation of the text.  

## TO-DO

 - Add word2vec and GloVe support
 - Add batch processing for BERT models
 - ~~Fix import scheme~~

License
----

MIT
