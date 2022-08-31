import json
import mmap
from tqdm import tqdm
import string
from .text_dataset import TextDataset, TextDatasetCache
from typing import Tuple, List
import os
import tarfile

import regex as re
import random

class CFQ(TextDataset):
    URL = "https://storage.cloud.google.com/cfq_dataset/cfq1.1.tar.gz"

    def tokenize_punctuation(self, text):
        # From https://github.com/google-research/google-research/blob/master/cfq/preprocess.py
        text = map(lambda c: ' %s ' % c if c in string.punctuation else c, text)
        return ' '.join(''.join(text).split())

    def preprocess_sparql(self, query):
        # From https://github.com/google-research/google-research/blob/master/cfq/preprocess.py
        """Do various preprocessing on the SPARQL query."""
        # Tokenize braces.
        query = query.replace('count(*)', 'count ( * )')

        tokens = []
        for token in query.split():
            # Replace 'ns:' prefixes.
            if token.startswith('ns:'):
                token = token[3:]
            # Replace mid prefixes.
            if token.startswith('m.'):
                token = 'm_' + token[2:]
            tokens.append(token)

        return ' '.join(tokens).replace('\\n', ' ')

    def load_data(self, fname: str) -> Tuple[List[str], List[str]]:
        # Split the JSON manually, otherwise it requires infinite RAM and is very slow.
        pin = "complexityMeasures".encode()
        offset = 1
        cnt = 0

        inputs = []
        outputs = []

        self.VARS = [f'?x{i}' for i in range(6)]

        with open(fname, "r") as f:
            data = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
            pbar = tqdm(total=len(data))
            pbar.update(offset)

            while True:
                pos = data.find(pin, offset+6)
                if pos < 0:
                    this = data[offset: len(data)-2]
                else:
                    this = data[offset: pos-5]
                    new_offset = pos - 4
                    pbar.update(new_offset - offset)
                    offset = new_offset
                d = json.loads(this.decode())
                inputs.append(self.tokenize_punctuation(d["questionPatternModEntities"]))
                #outputs.append(self.preprocess_sparql(d["sparqlPatternModEntities"]))
                outputs.append(self.get_perm_iso(self.preprocess_sparql(d["sparqlPatternModEntities"])))

                cnt += 1
                if pos < 0:
                    break

        return inputs, outputs

    def get_permutes(self, text):
        outputs = []
        wheres = re.findall('WHERE { .*? }', text)
        wheres = wheres[0]
        #remaining = text.replace(wheres,'')
        splits = wheres[8:-2].split(' . ')
        for i in range(self.permute_factor):
            outputs.append(text.replace(wheres, 'WHERE { ' + ' . '.join(splits) + ' }'))
            random.shuffle(splits) # shuffling after appending so permute_factor 1 = original dataset.
        return outputs

    def get_iso(self, text):
        #return text
        map_in = list(set(re.findall("\?x\d+", text)))
        temp = [f'@{i}' for i in range(len(map_in))]
        map_out = random.sample(self.VARS, len(map_in))
        for old_,new_ in zip(map_in, temp):
            text = text.replace(f' {old_} ', f' {new_} ')
        for old_,new_ in zip(temp, map_out):
            text = text.replace(f' {old_} ', f' {new_} ')
        return text

    def get_perm_iso(self, text):
        outputs = []
        outputs.extend(list(set(self.get_permutes(text)))) # could be fewer than max
        isoutputs = []
        for out in outputs:
            isoutputs.append(out) # iso_factor 1 is original only
            for i in range(self.iso_factor-1):
                isoutputs.append(self.get_iso(out))
        return list(set(isoutputs)) # could be fewer than max

    def build_cache(self) -> TextDatasetCache:
        index_table = {}

        if not os.path.isdir(os.path.join(self.cache_dir, "cfq")):
            gzfile = os.path.join(self.cache_dir, os.path.basename(self.URL))
            if not os.path.isfile(gzfile):
                assert False, f"Please download {self.URL} and place it in the {os.path.abspath(self.cache_dir)} "\
                               "folder. Google login needed."

            with tarfile.open(gzfile, "r") as tf:
                tf.extractall(path=self.cache_dir)

        splitdir = os.path.join(self.cache_dir, "cfq", "splits")
        for f in os.listdir(splitdir):
            if not f.endswith(".json"):
                continue

            name = f[:-5].replace("_split", "")
            with open(os.path.join(splitdir, f), "r") as f:
                ind = json.loads(f.read())

            index_table[name] = {
                "train": ind["trainIdxs"],
                "val": ind["devIdxs"],
                "test": ind["testIdxs"]
            }

        in_sentences, out_sentences = self.load_data(os.path.join(self.cache_dir, "cfq/dataset.json"))
        assert len(in_sentences) == len(out_sentences)
        return TextDatasetCache().build(index_table, in_sentences, out_sentences, split_punctuation=False,
                    permute_factor=self.permute_factor, iso_factor=self.iso_factor)
