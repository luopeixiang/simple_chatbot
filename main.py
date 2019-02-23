import torch
import torch.nn as nn
import torch.optim as optim

from utils import *
from model import EncoderRNN, Attn, AttnDecoder
from train import trainIters


#Configure models
model_name = 'cb_model'
attn_model = 'dot'
#attn_model = 'general'
#attn_model = 'concat'
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 64

voc, pairs = loadData('./datasets/conversations.csv')
n_tokens = voc.size

embedding = nn.Embedding(n_tokens, hidden_size)
encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
#attn = Attn(attn_model, hidden_size)
decoder = AttnDecoder(attn_model, embedding, hidden_size,
                        n_tokens, decoder_n_layers, dropout)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = encoder.to(device)
decoder = decoder.to(device)

# Configure training/optimization
clip = 50.0
teacher_forcing_ratio = 1.0
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_iteration = 4000
print_every = 1
save_every = 500
save_dir = "./checkpoints"
corpus_name = "movie conversations"
loadFilename = None

encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)

encoder.train()
decoder.train()
trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer,
    decoder_optimizer, embedding, encoder_n_layers, decoder_n_layers,
    save_dir, n_iteration, batch_size, print_every, save_every, clip,
    corpus_name, loadFilename)
