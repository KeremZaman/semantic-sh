# semantic-sh
[![PyPI version](https://badge.fury.io/py/semantic-sh.svg)](https://badge.fury.io/py/semantic-sh)
[![Actions Status](https://github.com/KeremZaman/semantic-sh/workflows/build/badge.svg)](https://github.com/KeremZaman/semantic-sh/actions)
[![PyPI download total](https://img.shields.io/pypi/dm/semantic-sh.svg)](https://pypi.python.org/pypi/semantic-sh/)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)

semantic-sh is a SimHash implementation to detect and group similar texts by taking power of word vectors and transformer-based language models such as BERT.

Documentation
=================

<!--ts-->
   * [Requirements](#requirements)
   * [Installation](#installing-via-pip)
   * [Usage](#usage)
        * [Use with BERT](#use-with-bert)
        * [Use with fasttext](#use-with-fasttext)
        * [Use with GloVe](#use-with-glove)
        * [Use with word2vec](#use-with-word2vec)
      * [Additional parameters](#additional-parameters)
      * [Text hashing](#hash-your-text)
      * [Add document](#add-document)
      * [Find similar documents](#find-similar)
      * [Calculate Hamming Distance](#get-hamming-distance-between-2-texts)
      * [Document groups](#go-through-all-document-groups)
      * [Save](#save-data)
      * [Load](#load-from-saved-file)
   * [API Server](#api-server)
      * [Installation](#installation)
      * [Standalone usage](#standalone-usage)
      * [Using with WSGI Container](#using-with-wsgi-container)
      * [API Reference](#api-reference)
      * [Docker](#with-docker)
   * [Some Implementation Details](#some-implementation-details)
<!--te-->


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

# Usage

```
from semantic_sh import SemanticSimHash
```

### Use with BERT:

```
sh = SemanticSimHash(model_type='bert-base-multilingual-cased', dim=768)
```

### Use with fasttext:
```
sh = SemanticSimHash(model_type='fasttext', dim=300, model_path='/path/to/cc.en.300.bin')
```

### Use with GloVe:
```
sh = SemanticSimHash(model_type='glove', dim=300, model_path='/path/to/glove.6B.50d.txt')
```

### Use with word2vec:

```
sh = SemanticSimHash(model_type='word2vec', dim=300, model_path='/path/to/en.w2v.txt')
```

### Additional parameters

Customize threshold (default:0) , hash length (default: 256-bit) and add stop words list.

```
sh = SemanticSimHash(model_type='fasttext', key_size=128, dim=300, model_path='pat_to_fasttext_vectors.bin', thresh=0.8, stop_words=['the', 'i', 'you', 'he', 'she', 'it', 'we', 'they'])
```

**Note:** BERT-based models do not require stop words list.

### Hash your text

```
sh.get_hash(['<your_text_0>', '<your_text_1>'])
```

### Add document

Add your document to the proper group

```
sh.add_document(['<your_text_0>', '<your_text_1>'])
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


```POST /api/hash```

Return hashes of given documents 

**Request Body**
```
{
    "documents": [
        "Here is the first document",
        "and second document"
    ]
}       
```

**Response Body**
```
{
    "hashes": [
        "0x7f636944d8c8",
        "0x5d134944428a4"
    ]
}    
```
***

```POST /api/add```

Add given documents and return hash and custom IDs of the documents

**Request Body**
```
{
    "documents": [
        "Here is the first document",
        "and second document"
    ]
}
        
```

**Response Body**
```
{
    "documents": [
        {
            "id": 1,
            "hash": 0x5d134944428a4"
        },
        {
            "id": 2,
            "hash": 0x7f636944d8c8"
        }
    ]
}     
```
***

```POST /api/find-similar```

Return similar documents to given text

**Request Body**
```
{
    "text": "Here is the text"
}       
```

**Response Body**
```
{
    "similar_texts": [
        "Here is the text",
        "First text here",
        "Here is text"
    ]
}    
```

*** 

```POST /api/distance```

Return Hamming distance between source and target texts

**Request Body**
```
{
    "src": "Here is the source text",
    "tgt": "Target text for measuring distance"
}       
```

**Response Body**
```
{
    "distance": 21
}    
```

***

```GET /api/similarity-groups```

Return buckets having more than one document ID

***

```GET /api/text/<int:id>```

Return the document according to its ID

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


License
----

MIT
