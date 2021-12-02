import json
import csv
import os

root_dir = '../../../'
vocab_dir = root_dir+'datasets/data_preprocessed/icews/vocab/'
dir = root_dir+'datasets/data_preprocessed/icews/'
os.makedirs(vocab_dir)
entity_vocab = {}
relation_vocab = {}
tim_vocab = {}
action_vocab = {}
entity_vocab['PAD'] = len(entity_vocab)
entity_vocab['UNK'] = len(entity_vocab)
relation_vocab['PAD'] = len(relation_vocab)
relation_vocab['DUMMY_START_RELATION'] = len(relation_vocab)
relation_vocab['NO_OP'] = len(relation_vocab)
relation_vocab['UNK'] = len(relation_vocab)
tim_vocab['PAD'] = len(tim_vocab)
tim_vocab['DUMMY_START_TIM'] = len(tim_vocab)
tim_vocab['NO_OP'] = len(tim_vocab)
tim_vocab['UNK'] = len(tim_vocab)
action_vocab['PAD'] = len(action_vocab)
action_vocab['DUMMY_START_ACTION'] = len(action_vocab)
action_vocab['NO_OP'] = len(action_vocab)
action_vocab['UNK'] = len(action_vocab)
entity_counter = len(entity_vocab)
relation_counter = len(relation_vocab)
tim_counter = len(tim_vocab)
action_counter = len(action_vocab)
for f in ['train.txt', 'dev.txt', 'test.txt', 'graph.txt']:
    with open(dir+f, encoding='UTF-8') as raw_file:
        csv_file = csv.reader(raw_file, delimiter='\t')
        for line in csv_file:
            e1,r,e2, tim = line
            if e1 not in entity_vocab:
                entity_vocab[e1] = entity_counter
                entity_counter += 1
            if e2 not in entity_vocab:
                entity_vocab[e2] = entity_counter
                entity_counter += 1
            if r not in relation_vocab:
                relation_vocab[r] = relation_counter
                relation_counter += 1
            if tim not in relation_vocab:
                tim_vocab[tim] = tim_counter
                tim_counter += 1
            if tim not in action_vocab:
                if r not in action_vocab:
                    action_vocab[r + ' ' + tim] = action_counter
                    action_counter += 1
with open(vocab_dir + 'entity_vocab.json', 'w') as fout:
    json.dump(entity_vocab, fout)
with open(vocab_dir + 'relation_vocab.json', 'w') as fout:
    json.dump(relation_vocab, fout)
with open(vocab_dir + 'tim_vocab.json', 'w') as fout:
    json.dump(tim_vocab, fout)
with open(vocab_dir + 'action_vocab.json', 'w') as fout:
    json.dump(action_vocab, fout)

