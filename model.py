
import torch
import torch.nn as nn
import  torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from data import SOS_token, EOS_token

class EncoderRNN(nn.Module):

    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.embedding = embedding
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                            bidirectional = True,
                            dropout = 0 if n_layers==1 else dropout)


    def forward(self, input_seq, input_lengths, hidden=None):

        embed = self.embedding(input_seq)
        try:
            padded_seq = pack_padded_sequence(embed, input_lengths)
        except:
            import pdb;pdb.set_trace()
        outputs, hidden = self.gru(padded_seq, hidden)
        outputs, _ = pad_packed_sequence(outputs)

        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]

        return outputs, hidden


class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        assert method in ["dot", "general", "concat"]
        self.method = method
        self.hidden_size = hidden_size

        if self.method == "general":
            self.attn = nn.Linear(hidden_size, hidden_size)
        elif self.method == "concat":
            self.attn = nn.Linear(2*hidden_size, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def dot(self, ht, hs):
        return torch.sum(ht * hs, dim=2)

    def general(self, ht, hs):
        hs_ = self.attn(hs)
        return torch.sum(ht * hs, dim=2)

    def concat(self, ht, hs):
        hs_ = self.attn(torch.cat([ht.expand(hs.size(0), -1, -1), hs], dim=2))
        return torch.sum(self.v * hs_, dim=2)

    def forward(self, hidden, encoder_outputs):

        attn_scores = getattr(self, self.method)(hidden, encoder_outputs)
        attn_scores = attn_scores.transpose(0, 1).unsqueeze(1)

        return F.softmax(attn_scores, dim=2)


class AttnDecoder(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size,
                    output_size, n_layers=1, dropout=0.1):

        super(AttnDecoder, self).__init__()
        self.n_layers = n_layers
        self.attn = Attn(attn_model, hidden_size)
        self.embedding = embedding
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                            dropout = 0 if n_layers==1 else dropout)

        self.fc = nn.Linear(hidden_size, output_size)
        self.concat = nn.Linear(2*hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)


    def forward(self, input_step, last_hidden, encoder_outputs):
        embed = self.embedding(input_step)

        try:
            rnn_output, hidden = self.gru(embed, last_hidden)
        except:
            import pdb;pdb.set_trace()
        attn_scores = self.attn(rnn_output, encoder_outputs)

        #attn_scores: (batch_size, 1, max_length)
        #encoder_outputs: (max_length, batch_size, hidden_size)
        context = attn_scores.bmm(encoder_outputs.transpose(0,1))
        #context: (batch_size, 1, hidden_size)

        #rnn_output: (1, batch_size, hidden_size)
        #Luong eq.5.
        context = context.squeeze(1)
        rnn_output = rnn_output.squeeze(0)
        ht_ = torch.tanh(self.concat(torch.cat([context, rnn_output], dim=1)))

        #eq.6.
        #output: batch_size, voc_size
        output = F.softmax(self.out(ht_), dim=1)

        return output, hidden



class GreedySearchDecoder(nn.Module):

    def __init__(self, encoder, decoder, voc):
        super(GreedySearchDecoder, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.voc = voc

    def forward(self, input_seq, input_lengths, max_length):

        #import pdb;pdb.set_trace()
        encoder_outputs, hidden = self.encoder(input_seq, input_lengths)
        decoder_hidden = hidden[:self.decoder.n_layers]
        decoder_input = torch.ones(1,1).long() * SOS_token

        all_words = []
        all_scores = []
        for t in range(max_length):
            decoder_out, decoder_hidden = self.decoder(
                                        decoder_input, decoder_hidden, encoder_outputs)

            word_score, word_idx = torch.max(decoder_out, dim=1)

            if word_idx.item() == EOS_token:
                break
            all_words.append(self.voc.idx2word[word_idx.item()])
            all_scores.append(word_score.item())
            decoder_input = word_idx.view(1, -1)

        return all_words, all_scores
