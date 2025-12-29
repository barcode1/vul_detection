import os
import torch
import numpy as np
import re
from transformers import AutoTokenizer
from gensim.models import Word2Vec, FastText
from typing import List, Dict, Any


class MultiEmbeddingTokenizer:
    def __init__(self, config: Dict[str, Any]):
        self.sec_bert_tokenizer = AutoTokenizer.from_pretrained(
            config['sec_bert']['model_name']
        )
        self.word2vec = None
        self.fasttext = None
        self.max_length = config['data']['max_seq_length']
        self.embedding_dim = config['embedding']['embedding_dim']

        # مسیر ذخیره‌سازی
        self.save_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'data',
            'embeddings'
        )

    def _simple_tokenize(self, text: str) -> List[str]:
        tokens = re.findall(r'\b\w+\b|[^\w\s]', text)
        return tokens if tokens else ['<PAD>']

    def train_word_embeddings(self, corpus: List[str]):
        """آموزش و ذخیره Word2Vec و FastText"""
        os.makedirs(self.save_dir, exist_ok=True)

        tokenized_corpus = [self._simple_tokenize(text) for text in corpus]

        self.word2vec = Word2Vec(
            sentences=tokenized_corpus,
            vector_size=self.embedding_dim,
            window=5,
            min_count=2,
            epochs=50,
            sg=1,
            workers=4
        )

        self.fasttext = FastText(
            sentences=tokenized_corpus,
            vector_size=self.embedding_dim,
            window=5,
            min_count=2,
            epochs=50,
            sg=1,
            workers=4
        )

        self.save_word_embeddings()

        print(f"✅ save models to : {self.save_dir}")

    def save_word_embeddings(self, path: str = None):
        """متد جدید برای ذخیره مدل‌ها

        Args:
            path: مسیر دلخواه برای ذخیره (اختیاری). اگر ندادید از self.save_dir استفاده می‌شود.
        """
        if self.word2vec is None or self.fasttext is None:
            raise ValueError("frist train models!")


        save_path = path if path is not None else self.save_dir
        os.makedirs(save_path, exist_ok=True)
        self.word2vec.save(os.path.join(save_path, 'word2vec_security.model'))
        self.fasttext.save(os.path.join(save_path, 'fasttext_security.model'))
        print(f"✅ saved models at: {save_path}")

    def load_word_embeddings(self, path: str = None):
        """بارگذاری مدل‌های آموزش‌دیده

        Args:
            path: مسیر دلخواه برای بارگذاری (اختیاری). اگر ندادید از self.save_dir استفاده می‌شود.
        """
        load_path = path if path is not None else self.save_dir

        w2v_path = os.path.join(load_path, 'word2vec_security.model')
        ft_path = os.path.join(load_path, 'fasttext_security.model')

        if os.path.exists(w2v_path) and os.path.exists(ft_path):
            self.word2vec = Word2Vec.load(w2v_path)
            self.fasttext = FastText.load(ft_path)
            print(f"✅ loaded models at: {load_path}")
        else:
            raise FileNotFoundError(f"models at  {load_path} not found!")

    def encode(self, texts: List[str]) -> Dict[str, Any]:
        return {
            'sec_bert': self._encode_sec_bert(texts),
            'word2vec': self._encode_word2vec(texts),
            'fasttext': self._encode_fasttext(texts)
        }

    def _encode_sec_bert(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        return self.sec_bert_tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
            return_attention_mask=True
        )

    def _encode_word2vec(self, texts: List[str]) -> torch.Tensor:
        embeddings = []
        for text in texts:
            tokens = self._simple_tokenize(text)
            token_vecs = [self.word2vec.wv[token] for token in tokens if token in self.word2vec.wv]
            embeddings.append(np.mean(token_vecs, axis=0) if token_vecs else np.zeros(self.embedding_dim))
        return torch.FloatTensor(np.array(embeddings))

    def _encode_fasttext(self, texts: List[str]) -> torch.Tensor:
        embeddings = []
        for text in texts:
            tokens = self._simple_tokenize(text)
            token_vecs = [self.fasttext.wv[token] for token in tokens]
            embeddings.append(np.mean(token_vecs, axis=0) if token_vecs else np.zeros(self.embedding_dim))
        return torch.FloatTensor(np.array(embeddings))