from typing import List, Optional, Generator, Tuple

import numpy as np

import string
import os
import errno
import pickle
import io

from transformers import BertModel, BertTokenizer
import fasttext


class SemanticSimHash(object):

    def __init__(self, model_type: str = 'fasttext', key_size: int = 256, dim: int = 300, stop_words: List[str] = [],
                 model_path: Optional[str] = None, thresh: int = 0):
        """
        Initialize projection matrix, model and buckets, assign given args to class members.

        :param model_type:_fasttext, word2vec, glove or any transformers model listed in Hugging Face model hub
        :param key_size: hash size (bits)
        :param dim: dimension of used embeddings
        :param stop_words: list of stop words to remove from the text (just for word embeddings e.g. fasttext,
        glove, word2vec)
        :param model_path: path of vector file for word embedding models
        :param thresh: threshold to decide the binary value to assign based on dot-product result
        """

        self._buckets = {}
        self._doc2id = {}
        self._documents = []
        self._type = ''

        self._thresh = thresh
        self.key_size = key_size
        self.dim = dim
        self._proj = self._create_proj(key_size, dim)
        self._stop_words = stop_words

        self._init_model(model_type, model_path)

    def _load_wordvec_model(self, emb_type: str, model_path: str):
        """
        Load word embeddings from file (glove, word2vec and fasttext vec file format)

        :param emb_type:
        :param model_path:
        :return:
        """
        model = {}

        fin = io.open(model_path, 'r', encoding='utf-8', newline='\n', errors='ignore')

        if emb_type in ['fasttext', 'word2vec']:
            fin.readline()  # pass first line containing vocab size and dimension info

        for line in fin:
            tokens = line.strip().split()
            word, vec = tokens[0], np.expand_dims(np.fromiter(map(float, tokens[1:]), dtype=float), 1)  # (d, 1)

            if vec.shape[0] != self.dim:
                raise Exception('Mismatched dimensions')

            model[word] = vec

        fin.close()

        return model

    def _init_model(self, model_type: str, model_path: Optional[str]):
        """
        Load pretrained model and tokenizer if required.

        :param model_type:
        :param model_path:
        :return:
        """

        print("Loading model...")

        if model_type in ['fasttext', 'glove', 'word2vec']:

            if model_path is None:
                raise Exception('To use word embeddings, model_path must be given.')
            elif not os.path.isfile(model_path):
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), model_path)

            if model_type == 'fasttext' and model_path.endswith('.bin'):
                self._model = fasttext.load_model(model_path)
                self._type = 'fasttext_bin'
            else:
                self._model = self._load_wordvec_model(model_type, model_path)
                self._type = 'wordvec'

        else:
            self._type = model_type
            self._tokenizer = BertTokenizer.from_pretrained(self._type)
            self._model = BertModel.from_pretrained(self._type)

    def _create_proj(self, key_size: int, dim: int) -> np.ndarray:
        """
        Return (key_size x dim) matrix with each column sampled from N(0,1).

        :param key_size:
        :param dim:
        :return:
        """

        return np.vstack((np.random.normal(0, 1, dim) for i in range(0, key_size)))

    def _get_wordvec_encoding(self, documents: List[str]) -> np.ndarray:
        """
        Remove stop words, take average of the vectors of the remaining words and return it as representation of text.

        :param documents:
        :return:
        """

        doc_tokens = []

        for doc in documents[:]:
            # remove punctuation and convert to lowercase
            doc = doc.translate(str.maketrans('', '', string.punctuation)).lower()

            # remove stop words
            tokens = [token for token in doc.split() if token not in self._stop_words]

            if len(tokens) == 0:
                raise Exception(f'Text is empty after filtering stop words and punctuation. The text was: {doc}')

            doc_tokens.append(tokens)

        # average of vectors of tokens
        doc_vecs = np.zeros((self.dim, len(doc_tokens)))

        if self._type == 'fasttext_bin':
            doc_vecs += np.array([np.average([self._model.get_word_vector(token) for token in doc], axis=0)
                                  for doc in doc_tokens]).T
        else:
            doc_vecs += np.array([np.average([self._model[token] for token in doc if token in self._model], axis=0)
                                  for doc in doc_tokens]).squeeze(axis=-1).T

        return doc_vecs

    def _get_bert_encoding(self, documents: List[str]) -> np.ndarray:
        """
        Tokenize text, truncate to max length of tokens specified by model, encode and return embedding of CLS token as
        representation of the text

        :param documents:
        :return:
        """

        input_ids = self._tokenizer(documents, padding=True, truncation=True, return_tensors='pt')['input_ids']
        last_layer_emb = self._model(input_ids)[0][:, 0, :].squeeze(1).t()
        doc_embs = last_layer_emb.detach().numpy()  # return CLS token embeddings as numpy array

        return doc_embs

    def _get_encoding(self, documents: List[str]) -> np.ndarray:
        """
        Call proper function to get text encoding according to model type

        :param documents:
        :return:
        """

        if self._type in ['fasttext_bin', 'wordvec']:
            return self._get_wordvec_encoding(documents)
        else:
            return self._get_bert_encoding(documents)

    def save(self, fname: str):
        """
        Dump all class members to file

        :param fname:
        :return:
        """

        with open(fname, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(fname: str) -> 'SemanticSimHash':
        """
        Load serialized state

        :param fname:
        :return:
        """

        with open(fname, 'rb') as f:
            obj = pickle.load(f)

        return obj

    def get_hash(self, documents: List[str]) -> List[int]:
        """
        Calculate and return simhashes of given documents

        Encode text, multiply with projection matrix, create hash and return it.
        :param documents:
        :return:
        """

        enc = self._get_encoding(documents)
        y = np.matmul(self._proj, enc)
        b = np.where(y <= self._thresh, 0, 1)  # assign binary values wrt threshold
        hash_vals = [b_vec.dot(2 ** np.arange(b_vec.size)[::-1]) for b_vec in b.T]  # convert binary array to integer

        return hash_vals

    def add_document(self, documents: List[str]) -> Tuple[List[int], List[int]]:
        """
        Hash text, add to its bucket, return hash of text.

        :param documents:
        :return:
        """

        hashes = self.get_hash(documents)
        ids = []

        for h, txt in zip(hashes, documents):

            if txt not in self._doc2id:
                self._doc2id[txt] = len(self._documents)
                self._documents.append(txt)

            ids.append(self._doc2id[txt])

            if h in self._buckets:
                self._buckets[h].append(self._doc2id[txt])
            else:
                self._buckets[h] = [self._doc2id[txt]]

        return hashes, ids

    def find_similar(self, txt: str) -> List[str]:
        """
        Hash text and return all texts inside that bucket.

        :param txt:
        :return:
        """
        h = self.get_hash([txt])[0]
        if h in self._buckets:
            return [self._documents[id] for id in self._buckets[h]]
        else:
            return []

    def get_distance(self, txt0: str, txt1: str) -> int:
        """
        Return hamming distance of the hashes of given text pair.

        :param txt0:
        :param txt1:
        :return:
        """
        diff = self.get_hash([txt0])[0] ^ self.get_hash([txt1])[0]
        diff &= (1 << self.key_size) - 1  # get 2's complement representation of negative numbers
        dist = sum(map(int, bin(diff)[2:]))  # [2:] to exclude '0b' part
        return dist

    def get_similar_groups(self) -> Generator[List[str], None, None]:
        """
        Return buckets that have more than one document
        :return:
        """
        for group in self._buckets.values():
            if len(group) > 1:
                yield group

    def get_doc_by_id(self, doc_id: int):
        """
        Return document with the given id

        :param doc_id:
        :return:
        """
        return self._documents[doc_id]
