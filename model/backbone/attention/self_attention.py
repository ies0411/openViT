import torch
import torch.nn as nn


class MultiheadedSelfAttention(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_heads = model_cfg.num_heads
        self.embedding_dimension = model_cfg.embedding_dimension
        # self.head_num = model_cfg.head_num
        self.head_dimension = int(self.embedding_dimension / self.num_heads)
        self.query = nn.Linear(self.embedding_dimension, self.embedding_dimension)
        self.key = nn.Linear(self.embedding_dimension, self.embedding_dimension)
        self.value = nn.Linear(self.embedding_dimension, self.embedding_dimension)
        self.dropout = nn.Dropout(self.model_cfg.drop_rate)
        self.scale = torch.sqrt(self.head_dimension*torch.ones(1)).to(self.device)

    def forward(self, x):
        # x: batch_size, patch개수 + 1, embedding_dimension
        batch_size = x.size(0)
        q = self.query(x)  # q : batch_size, patch개수 + 1, embedding_dimension
        k = self.key(x)  # k : batch_size, patch개수 + 1, embedding_dimension
        v = self.value(x)  # v : batch_size, patch개수 + 1, embedding_dimension
        q = q.view(batch_size, -1, self.num_heads, self.head_dimension).permute(0,2,1,3) # batch_size, patch개수 + 1 ,head number, head dimension -> B, head_number, patch개수 + 1, head_dimension(int(embedding_dimension / num_heads)
        k = k.view(batch_size, -1, self.num_heads, self.head_dimension).permute(0,2,3,1) # batch_size, patch개수 + 1 ,head number, head dimension -> B, head_number, head_dimension(int(embedding_dimension / num_heads), patch개수 + 1
        v = v.view(batch_size, -1, self.num_heads, self.head_dimension).permute(0,2,1,3) # batch_size, patch개수 + 1 ,head number, head dimension -> B, head_number, patch개수 + 1, head_dimension(int(embedding_dimension / num_heads)
        attention = torch.softmax(q @ k / self.scale, dim=-1) # batch_size, head_number, patch개수 + 1, patch개수 + 1
        x = self.dropout(attention) @ v # batch_size, num_heads,  patch개수 + 1,  patch개수 + 1 x  B, num_heads, patch개수 + 1, head_dimension => B, num_heads, patch개수 + 1, head_dimension
        x = x.permute(0,2,1,3).reshape(batch_size, -1, self.embedding_dimension) #batch_size, patch개수 + 1, num_heads, head_dimension -> B, patch개수 + 1 ,embedding_dimension(num_heads x head_dimension)
