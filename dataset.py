import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from transformers import BertTokenizer
from util import load_balls, preprocess_data

class nballDataset(Dataset):
    def __init__(self, data, nball, model_url, max_length=512):

        # Load and preprocess the dataset
        self.data = data
        self.nball = nball
        self.model_url = model_url
        # Train data with exist nball:
        self.data = self.data[self.data['formatted_sense_id'].isin(self.nball.keys())].copy()

        # Process sense indices and embeddings
        # self.sense_labels = list(self.nball.keys())
        # self.embeddings = [self.nball[label].center for label in self.sense_labels]
        # self.embeddings = torch.tensor(np.array(self.embeddings), dtype=torch.float32)
        # self.norms = [self.nball[label].distance for label in self.sense_labels]
        # self.norms = torch.tensor(np.array(self.norms), dtype=torch.float32)

        self.max_length = max_length
        # Initialize the tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(self.model_url)
        # Tokenize the data
        self.tokenize_data()

    def tokenize_data(self):
        # Tokenize all sentences in the dataset
        print("Tokenizing sentences...")
        tokenized_data = self.tokenizer(list(self.data['sentence_text']), padding=True, truncation=True, return_tensors="pt", max_length=self.max_length)
        print("Tokenizing finished.")
        self.data['input_ids'] = tokenized_data['input_ids'].tolist()
        self.data['attention_mask'] = tokenized_data['attention_mask'].tolist()
        # Find word indices
        print("Calculating word indices...")
        self.data['word_index'] = [self.find_word_index(sentence_ids, word) for sentence_ids, word in zip(self.data['input_ids'], self.data['word'])]
        print('Tokenizing finished.')

    def find_word_index(self, sentence_ids, word):
        # Locate the start index of the word in the tokenized sentence
        word_tokens = self.tokenizer.tokenize(word)
        word_ids = self.tokenizer.convert_tokens_to_ids(word_tokens)
        for i in range(len(sentence_ids) - len(word_tokens) + 1):
            if sentence_ids[i:i+len(word_tokens)] == word_ids:
                return i
        return -1

    def get_data(self):
        return self.data.copy()
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Retrieve tensor data for each item
        item = self.data.iloc[idx]
        input_ids = torch.tensor(item['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(item['attention_mask'], dtype=torch.long)
        word_index = torch.tensor(item['word_index'], dtype=torch.long)
        sense_idx = torch.tensor(item['sense_idx'], dtype=torch.long)
        lemma_idx = torch.tensor(item['lemma_idx'], dtype=torch.long)
        overall_idx = torch.tensor(idx, dtype=torch.long)
        # group_idces = torch.tensor(item['sense_group'], dtype=torch.long)
        # nball_embedding = self.embeddings[sense_idx]
        # norm = self.norms[sense_idx]
        # lemma = item['lemma']
        # group_embeddings = self.embeddings[group_idces]
        # group_norms = self.norms[group_idces]
        
        return input_ids, attention_mask, word_index, sense_idx, lemma_idx, overall_idx