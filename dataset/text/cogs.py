import os
from framework.utils import download
import csv
from .typed_text_dataset import TypedTextDataset, TypedTextDatasetCache
from ..sequence import TypedTextSequenceTestState

import regex as re
import random

class COGS(TypedTextDataset):
    URL_BASE = "https://raw.githubusercontent.com/najoungkim/COGS/main/data/"
    SPLT_TYPES = ["train", "test", "valid", "gen"]
    NAME_MAP = {"valid": "dev"}

    def build_cache(self) -> TypedTextDatasetCache:

        types = []
        type_list = []
        type_map = {}

        index_table = {}
        in_sentences = []
        out_sentences = []

        if self.permute_factor != 1:
            raise NotImplementedError

        self.VARS = [f'{i}' for i in range(27)]

        for st in self.SPLT_TYPES:
            fname = self.NAME_MAP.get(st, st) + ".tsv"
            split_fn = os.path.join(self.cache_dir, fname)
            os.makedirs(os.path.dirname(split_fn), exist_ok=True)

            full_url = self.URL_BASE + fname
            print("Downloading", full_url)
            download(full_url, split_fn, ignore_if_exists=True)

            index_table[st] = []

            with open(split_fn, "r") as f:
                d = csv.reader(f, delimiter="\t")
                for line in d:
                    i, o, t = line

                    index_table[st].append(len(in_sentences))
                    in_sentences.append(i)
                    #out_sentences.append(o)
                    out_sentences.append(self.get_perm_iso(o))

                    tind = type_map.get(t)
                    if tind is None:
                        type_map[t] = tind = len(type_list)
                        type_list.append(t)

                    types.append(tind)

                assert len(in_sentences) == len(out_sentences)

        return TypedTextDatasetCache().build({"default": index_table}, in_sentences, out_sentences, types, type_list)

    def get_permutes(self, text):
        return [text]

    def get_iso(self, text):
        #return text
        map_in = list(set(re.findall(" \d+ ", text)))
        temp = [f' @{i} ' for i in range(len(map_in))]
        map_out = random.sample(self.VARS, len(map_in))
        for old_,new_ in zip(map_in, temp):
            text = text.replace(f'{old_}', f'{new_}')
        for old_,new_ in zip(temp, map_out):
            text = text.replace(f'{old_}', f' {new_} ')
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


    def start_test(self) -> TypedTextSequenceTestState:
        return TypedTextSequenceTestState(lambda x: " ".join(self.in_vocabulary(x)),
                                          lambda x: " ".join(self.out_vocabulary(x)),
                                          self._cache.type_names)
