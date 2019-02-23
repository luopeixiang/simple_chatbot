import itertools

import torch

from data import *

def indexedFromSentence(sent_batch, voc):
    #lengths = [len(sent.split()) for sent in sent_batch]
    indexes_batch = [[voc.word2idx[word] for word in sent.split()]
                        for sent in sent_batch]
    lengths = [len(indexes) for indexes in indexes_batch]
    return indexes_batch, lengths

def zeroPadding(indexes_batch, voc):
    padded = list(itertools.zip_longest(*indexes_batch, fillvalue=PAD_token))
    return torch.LongTensor(padded)

def binaryMatrix(padded_batch, voc):
    pad_matrix = torch.ones_like(padded_batch) * PAD_token
    return torch.ByteTensor((pad_matrix != padded_batch))

def inputVar(sent_batch, voc):
    indexes_batch, lengths = indexedFromSentence(sent_batch, voc)
    padded_indexes_batch = zeroPadding(indexes_batch, voc)

    return padded_indexes_batch, torch.tensor(lengths)

def outputVar(sent_batch, voc):
    indexes_batch, lengths = indexedFromSentence(sent_batch, voc)
    max_target_len = max(lengths)
    padded_batch = zeroPadding(indexes_batch, voc)
    mask = binaryMatrix(padded_batch, voc)
    return padded_batch, mask, max_target_len

def batch2TrainData(pair_batch, voc):
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, target_batch = list(zip(*pair_batch))
    input_var, input_lengths = inputVar(input_batch, voc)
    target_var, mask, max_target_len = outputVar(target_batch, voc)

    return input_var, input_lengths, target_var, mask, max_target_len

def loadData(path):
    voc = Voc()
    pairs = []
    with open(path, 'r') as f:
        for line in f:
            pair = line.strip('\n').split('\t')
            pairs.append(pair)
            pair[0] += " <eos>"
            pair[1] += " <eos>"
            voc.addSentence(pair[0])
            voc.addSentence(pair[1])
    pairs.sort(key=lambda pair:len(pair[0]), reverse=True)
    return voc, pairs

def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    cross_entropy = -torch.log(torch.gather(inp, 1, target.view(-1,1)).squeeze(1))
    loss = cross_entropy.masked_select(mask).mean()
    return loss, nTotal.item()

if __name__ == "__main__":
    voc, pairs = loadData("./datasets/conversations.csv")
    sample_pairs = pairs[:10]
    input_var, input_lengths, target_var, mask, max_target_len = batch2TrainData(sample_pairs, voc)
