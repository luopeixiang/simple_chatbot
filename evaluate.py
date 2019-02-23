import re

import torch
import torch.nn as nn

from model import EncoderRNN, AttnDecoder, GreedySearchDecoder
from utils import *
from data import Voc

#load Model and interact with user
model_path = "./checkpoints/cb_model/2-2_500/4000_checkpoint.tar"

def preprocess_input(s):
    s = s.lower().strip()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    s += " <eos>"
    return s



MAX_LENGTH = 10
def evalute(searcher, sentence, max_length=MAX_LENGTH):
    #返回decoded words

    sentence = preprocess_input(sentence)
    input_seq, length = inputVar([sentence], searcher.voc)
    all_words, _ = searcher(input_seq, length, max_length)

    return " ".join(all_words)



def evaluateInput(searcher):
    #处理与用户的交互
    while True:
        try:
            input_sentence = input("Input> ")
            if input_sentence == "q":
                break
            out_sentence = evalute(searcher, input_sentence)
            print("Chat_bot> ", out_sentence)
        except:
            print("Unknown Word in input, try again")

#model Configure
model_name = 'cb_model'
attn_model = 'dot'
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 64
loadFilename = None
checkpoint_iter = 4000



#load model

print("Load checkpint...")
checkpoint = torch.load(model_path)
encoder_sd = checkpoint['en']
decoder_sd = checkpoint['de']
embedding_sd = checkpoint['embedding']
voc = Voc()
voc.__dict__ = checkpoint['voc_dict']

size = voc.size

embedding = nn.Embedding(size, hidden_size)
embedding.load_state_dict(embedding_sd)
encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
encoder.load_state_dict(encoder_sd)
decoder = AttnDecoder(attn_model, embedding, hidden_size, size, decoder_n_layers, dropout)
decoder.load_state_dict(decoder_sd)

searcher = GreedySearchDecoder(encoder, decoder, voc)
evaluateInput(searcher)
