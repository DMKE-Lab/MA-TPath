import json
import csv
import os
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
