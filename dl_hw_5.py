import torch
import nltk
from nltk.corpus import conll2000

torch.manual_seed(1)

nltk.download('conll2000')

train_data = conll2000.chunked_sents('train.txt')

train_sents = [[(word, tag) for word, tag, chunk in nltk.chunk.tree2conlltags(sent)] for sent in train_data]

acc_train = []
for item in train_sents:
    wwords = []
    ttags = []
    for ttuple in item:
        wwords.append(ttuple[0])
        ttags.append(ttuple[1])
    acc_train.append((wwords, ttags))


word_to_ix = {}
for sent, tags in acc_train:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)


tag_to_ix = {}
for sent, tags in acc_train:
    for tag in tags:
        if tag not in tag_to_ix:
            tag_to_ix[tag] = len(tag_to_ix)


EMBEDDING_DIM = 100
HIDDEN_DIM = 100

class LSTMTagger(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_layer = torch.nn.Embedding(len(word_to_ix), EMBEDDING_DIM)     
        self.lstm = torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM)
        
        self.pos_predictor = torch.nn.Linear(HIDDEN_DIM, len(tag_to_ix)) 

    def forward(self, token_ids):
        embeds = self.embedding_layer(token_ids)
        lstm_out, _ = self.lstm(embeds.view(len(token_ids), 1, -1))
        logits = self.pos_predictor(lstm_out.view(len(token_ids), -1))
        proba = torch.nn.functional.softmax(logits, dim=1)

        return proba
    

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)
    

model = LSTMTagger()
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


for epoch in range(10):
    for sentence, tags in acc_train:
        model.zero_grad()

        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)

        tag_scores = model(sentence_in)

        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()
    print(loss)


with torch.no_grad():
    inputs = prepare_sequence(acc_train[0][0], word_to_ix)
    tag_scores = model(inputs)

    print(tag_scores)