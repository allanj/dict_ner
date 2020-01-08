# 
# @author: Allan
#

from tqdm import tqdm
from common import Sentence, Instance
from typing import List
import re

class Reader:

    def __init__(self, digit2zero:bool=True):
        """
        Read the dataset into Instance
        :param digit2zero: convert the digits into 0, which is a common practice for LSTM-CRF.
        """
        self.digit2zero = digit2zero
        self.vocab = set()

    def read_txt(self, file: str, number: int = -1) -> List[Instance]:
        print("Reading file: " + file)
        insts = []
        with open(file, 'r', encoding='utf-8') as f:
            words = []
            labels = []
            for line in tqdm(f.readlines()):
                line = line.rstrip()
                if line == "":
                    insts.append(Instance(Sentence(words), labels))
                    words = []
                    labels = []
                    if len(insts) == number:
                        break
                    continue
                word, label = line.split()
                if self.digit2zero:
                    word = re.sub('\d', '0', word) # replace digit with 0.
                words.append(word)
                self.vocab.add(word)
                labels.append(label)
        print("number of sentences: {}".format(len(insts)))
        return insts

    def read_conll(self, file: str, number: int = -1, is_train: bool = True) -> List[Instance]:
        print("Reading file: " + file)
        insts = []
        num_entity = 0
        with open(file, 'r', encoding='utf-8') as f:
            words = []
            heads = []
            deps = []
            labels = []
            tags = []
            for line in tqdm(f.readlines()):
                line = line.rstrip()
                if line == "":
                    insts.append(Instance(Sentence(words, tags), labels))
                    words = []
                    heads = []
                    deps = []
                    labels = []
                    tags = []
                    if len(insts) == number:
                        break
                    continue
                vals = line.split()
                word = vals[1]
                head = int(vals[6])
                dep_label = vals[7]
                pos = vals[3]
                label = vals[10]
                if self.digit2zero:
                    word = re.sub('\d', '0', word)  # replace digit with 0.
                words.append(word)
                heads.append(head - 1)  ## because of 0-indexed.
                deps.append(dep_label)
                tags.append(pos)
                self.vocab.add(word)
                labels.append(label)
                if label.startswith("B-"):
                    num_entity += 1
        print("number of sentences: {}, number of entities: {}".format(len(insts), num_entity))
        return insts



