from utils import *
from dataloader import dec_vocab_size


class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.attention1 = MutiHeadAttention()
        self.attention2 = MutiHeadAttention()
        self.fc = PositionWiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_masks, cro_masks):
        dec_outputs, attention1 = self.attention1(dec_inputs, dec_inputs, dec_inputs, dec_masks)
        dec_outputs, attention2 = self.attention2(dec_outputs, enc_outputs, enc_outputs, cro_masks)
        outputs = self.fc(dec_outputs)
        return outputs, attention1, attention2


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.word_emb = nn.Embedding(dec_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        outputs = self.word_emb(dec_inputs)
        outputs = self.pos_emb(outputs.transpose(0, 1)).transpose(0, 1).cuda()
        pad_masks = pad_mask(dec_inputs, dec_inputs).cuda()
        seq_masks = seq_mask(dec_inputs).cuda()
        dec_masks = torch.gt((pad_masks + seq_masks), 0).to(device)
        cro_masks = pad_mask(dec_inputs, enc_inputs)
        weights1 = []
        weights2 = []
        for layer in self.layers:
            outputs, weight1, weight2 = layer(outputs, enc_outputs, dec_masks, cro_masks)
            weights1.append(weight1)
            weights2.append(weight2)
        return outputs, weights1, weights2
