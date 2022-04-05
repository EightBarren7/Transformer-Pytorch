from utils import *
from dataloader import enc_vocab_size


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.attention = MutiHeadAttention()
        self.fc = PositionWiseFeedForwardNet()

    def forward(self, inputs, pad_masks):
        outputs, weight = self.attention(inputs, inputs, inputs, pad_masks)
        outputs = self.fc(outputs)
        return outputs, weight


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.word_emb = nn.Embedding(enc_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, inputs):
        weights = []
        outputs = self.word_emb(inputs)
        outputs = self.pos_emb(outputs.transpose(0, 1)).transpose(0, 1)
        pad_masks = pad_mask(inputs, inputs)
        for layer in self.layers:
            outputs, weight = layer(outputs, pad_masks)
            weights.append(weight)
        return outputs, weights
