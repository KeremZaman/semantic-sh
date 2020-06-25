# semantic-sh


semantic-sh is a SimHash implementation to detect and group similar texts by taking power of word vectors and transformer-based language models (BERT).



# Requirements
 - fasttext
 - transformers
 - pytorch
 - numpy
# Installing via pip

```sh
$ pip install semantic_sh
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

# Some Implementation Details

This is a simplified implementation of simhash by just creating random 
vectors and assigning 1 or 0 according to the result of dot product of each of these vectors with
represantation of the text.  

## TO-DO

 - Add word2vec and GloVe support
 - Add batch processing for BERT models

License
----

MIT
