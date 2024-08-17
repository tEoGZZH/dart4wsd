import torch
import torch.nn as nn
from transformers import BertModel

class nballBertNorm(nn.Module):
    def __init__(self, model_url, output_dim):
        super(nballBertNorm, self).__init__()
        self.model_url = model_url
        self.bert = BertModel.from_pretrained(self.model_url)
        self.projection = nn.Linear(self.bert.config.hidden_size, output_dim)
        # Direct regression for norm
        self.regressor = nn.Linear(output_dim, 1)
        
    def forward(self, input_ids, attention_mask, word_index):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        word_embeddings = hidden_states[range(len(word_index)), word_index]
        projected_output = self.projection(word_embeddings)
        norm = self.regressor(projected_output)
        return norm


class nballBertDirection(nn.Module):
    def __init__(self, model_url, output_dim):
        super(nballBertDirection, self).__init__()
        self.model_url = model_url
        self.bert = BertModel.from_pretrained(self.model_url)
        self.projection = nn.Linear(self.bert.config.hidden_size, output_dim)
        
    def forward(self, input_ids, attention_mask, word_index):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        word_embeddings = hidden_states[range(len(word_index)), word_index]
        projected_output = self.projection(word_embeddings)
        return projected_output