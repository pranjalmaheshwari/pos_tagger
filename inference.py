import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import time

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class tagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(tagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim).cuda()),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim).cuda()))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space)
        return tag_scores


if(len(sys.argv) < 4):
	print 'No path given'
	exit()

def read_data(path, sentence='_sentences.txt', tag='_tags.txt'):
	f_sentence = open(path+sentence, "r")
	f_tag = open(path+tag, "r")
	ret = map( lambda x: (x[0].strip().split(), x[1].strip().split()), zip(f_sentence.readlines(), f_tag.readlines()))
	f_sentence.close()
	f_tag.close()
	return ret

def to_idx(data):
	_to_ix = {}
	for sent in training_data:
		for word in sent:
			if word not in _to_ix:
				_to_ix[word] = len(_to_ix)
	return _to_ix

def prepare_sequence(seq, to_ix):
	idxs = [to_ix[w] if w in to_ix else len(to_ix) for w in seq]
	tensor = torch.LongTensor(idxs)
	return autograd.Variable(tensor)

def accuracy(data, model, word_to_ix, tag_to_ix):
    correct = 0
    total = 0
    i = 0
    for sentence, tags in data:
        model.hidden = model.init_hidden()
        inputs = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)
        tag_scores = model(inputs.cuda())
        _, predictions = torch.max(tag_scores, 1)
        correct += (predictions == targets.cuda()).sum().data[0]
        total += targets.size(0)
    return (100.0*correct)/total


train_path = sys.argv[1]
val_path = sys.argv[2]

training_data = read_data(train_path)

word_to_ix = {}
tag_to_ix = {}
for sent, tags in training_data:
	for word in sent:
		if word not in word_to_ix:
			word_to_ix[word] = len(word_to_ix)
	for word in tags:
		if word not in tag_to_ix:
			tag_to_ix[word] = len(tag_to_ix)

val_data = read_data(val_path)

print "Data read"
sys.stdout.flush()

EMBEDDING_DIM = 512
HIDDEN_DIM = 512
epochs = 10
models_saved = 0
output_file_path = 'lstm'

# model = tagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix)+1, len(tag_to_ix))
# model.cuda()
model = torch.load(sys.argv[3])

start_time = time.time()
print_time = 0.0

print "Training started with data size:", len(training_data)
sys.stdout.flush()

accuracy_train = accuracy(training_data, model, word_to_ix, tag_to_ix)
accuracy_val = accuracy(val_data, model, word_to_ix, tag_to_ix)
print('train:, %.4f, validation:, %.4f' %
            (accuracy_train, accuracy_val))
sys.stdout.flush()

print 'Finished testing with time per sample :', (time.time()-start_time)/(len(training_data)+len(val_data)), 'secs'
sys.stdout.flush()
