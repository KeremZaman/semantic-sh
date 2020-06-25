from typing import List, Optional, Generator

import numpy as np
import torch

import string
import os
import errno

from transformers import BertModel, BertTokenizer
import fasttext


class SemanticSimHash(object):

    def __init__(self, model_type: str = 'fasttext', key_size: int = 256, dim: int = 300, stop_words: List[str] = [],
                 model_path: Optional[str] = None, thresh: int = 0):
        """Initialize projection matrix, model and buckets, assign given args to class members."""

        self._buckets = {}

        self._type = model_type

        self._thresh = thresh
        self._proj = self._create_proj(key_size, dim)
        self._stop_words = stop_words

        self._init_model(model_path)

    def _init_model(self, model_path: Optional[str]):
        """Load pretrained model and tokenizer if required."""

        print("Loading model...")

        if self._type == 'fasttext':
            if model_path is None:
                raise Exception('To use fasttext embeddings, model_path must be given.')
            elif not os.path.isfile(model_path):
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), model_path)

            self._model = fasttext.load_model(model_path)

        else:
            self._tokenizer = BertTokenizer.from_pretrained(self._type)
            self._model = BertModel.from_pretrained(self._type)

    def _create_proj(self, key_size: int, dim: int) -> np.ndarray:
        """"Return (key_size x dim) matrix with each column sampled from N(0,1)."""

        return np.vstack((np.random.normal(0, 1, dim) for i in range(0, key_size)))

    def _get_fasttext_encoding(self, txt: str) -> np.ndarray:
        """Remove stop words, take average of the vectors of the remaining words and return it as representation of text."""

        # remove punctuation and convert to lowercase
        txt = txt.translate(str.maketrans('', '', string.punctuation)).lower()

        # remove stop words
        tokens = [token for token in txt.split() if token not in self._stop_words]

        if len(tokens) == 0:
            raise Exception(f'Text is empty after filtering stop words and punctuation. The text was: {txt}')

        # average of vectors of tokens
        doc_vec = np.average([self._model.get_word_vector(token) for token in tokens], axis=0)

        return doc_vec

    def _get_bert_encoding(self, txt: str) -> np.ndarray:
        """Tokenize text, truncate to 512 tokens, encode and return embedding of CLS token as representation of the text"""
        input_ids = torch.tensor(self._tokenizer.encode(txt, max_length=512)).unsqueeze(0)  # Batch size 1
        last_layer_emb = self._model(input_ids)[0]
        doc_vec = last_layer_emb[0][0].detach().numpy()  # return CLS token embedding as numpy array

        return doc_vec

    def _get_encoding(self, text: str) -> np.ndarray:
        """Call proper function to get encoding of text according to model type"""
        if self._type == 'fasttext':
            return self._get_fasttext_encoding(text)
        else:
            return self._get_bert_encoding(text)

    def get_hash(self, txt: str) -> int:
        """Encode text, multiply with projection matrix, create hash and return it."""
        enc = self._get_encoding(txt)
        y = np.matmul(self._proj, enc)
        b = np.where(y <= self._thresh, 0, 1)  # assign binary values wrt threshold
        hash_val = b.dot(2 ** np.arange(b.size)[::-1])  # convert binary array to integer

        return hash_val

    def add_document(self, txt: str) -> None:
        """Hash text and add to its bucket."""
        h = self.get_hash(txt)

        if h in self._buckets:
            self._buckets[h].append(txt)
        else:
            self._buckets[h] = [txt]

    def find_similar(self, txt: str) -> List[str]:
        """Hash text and return all texts inside that bucket."""
        return self._buckets[self.get_hash(txt)]

    def get_distance(self, txt0: str, txt1: str) -> int:
        """Return hamming distance of the hashes of given text pair."""
        diff = self.get_hash(txt0) ^ self.get_hash(txt1)
        dist = sum(map(int, bin(diff)[2:]))  # [2:] to exclude '0b' part
        return dist

    def get_similar_groups(self) -> Generator[List[str], None, None]:
        """Return buckets that have more than one document"""
        for group in self._buckets.values():
            if len(group) > 1:
                yield group
