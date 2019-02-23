PAD_token = 0
SOS_token = 1
EOS_token = 2

class Voc(object):
    def __init__(self):
        self.word2idx = {"<pad>":0, "<sos>":1, "<eos>":2}
        self.idx2word = {0:"<pad>", 1:"<sos>", 2:"<eos>"}
        self.size = 3


    def addWord(self, word):

        if word not in self.word2idx:
            self.word2idx[word] = self.size
            self.idx2word[self.size] = word
            self.size += 1

    def addSentence(self, sent):
        for word in sent.split():
            self.addWord(word)
