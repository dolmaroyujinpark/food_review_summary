import torch.nn as nn
import torch

class FeedForward(nn.Module):
    def __init__(self, embed_size, forward_expansion):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, forward_expansion * embed_size)
        self.fc2 = nn.Linear(forward_expansion * embed_size, embed_size)

    def forward(self, x):
        #print("FeedForward Input:", x.shape)
        x = torch.relu(self.fc1(x))
        #print("After FC1 and ReLU:", x.shape)
        x = self.fc2(x)
        #print("FeedForward Output:", x.shape)
        return x