from encoder import Encoder
from decoder import Decoder
from torch import nn
from config import *
from dataloader import dec_vocab_size


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder().to(device)
        self.decoder = Decoder().to(device)
        self.fc = nn.Linear(d_model, dec_vocab_size, bias=False).to(device)

    def forward(self, enc_inputs, dec_inputs):
        """
        enc_inputs: [batch_size, en_maxlen]
        dec_inputs: [batch_size, cn_maxlen-1]
        return: [batch_size * (cn_maxlen-1), cn_vocab_size]
        """
        enc_outputs, enc_attention = self.encoder(enc_inputs)
        # enc_outputs: [batch_size, en_maxnlen, d_model]
        dec_outputs, dec_attention1, dec_attention2 = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        # dec_outputs: [batch_size, cn_maxlen-1, d_model]
        outputs = self.fc(dec_outputs)
        # outputs: [batch_size, cn_maxlen-1, cn_vocab_size]
        return outputs.view(-1, outputs.size(-1)), enc_attention, dec_attention1, dec_attention2

    def encode(self, enc_inputs):
        outputs, _ = self.encoder(enc_inputs)
        return outputs

    def decode(self, dec_inputs, enc_inputs, enc_outputs):
        dec_outputs, _, _ = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        outputs = self.fc(dec_outputs)
        return outputs
