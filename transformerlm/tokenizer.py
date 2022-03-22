class Tokenizer: 
    "tokenize given input"
    def __init__(self, mode='ext_dict', data=None, extdict=None): 
        self.mode = mode 
        if mode == 'use_all': 
            assert data is not None, f"for mode {mode}, data is required"
            assert extdict is None 
            self.build_vocab_all_words(data)
            self.tokenize_data = self.tokenize_data_all_words
        elif mode == 'ext_dict': 
            assert data is None
            assert extdict is not None, f"for mode {mode}, external dict is required"
            self.build_vocab_ext_dict(extdict)
            self.tokenize_data = self.tokenize_data_all_words
        else: 
            raise Exception(f"Unknown mode : {mode}")
    
    def build_vocab_ext_dict(self, extdict): 
        # ext dict must be sorted
        # special symbols match order of fairseq vocab
        # see: https://github.com/pytorch/fairseq/blob/1479d311d5/fairseq/data/dictionary.py
        self.word2id = {
            '<bos>': 0, 
            '<pad>': 1, 
            '<eos>': 2, 
            '<unk>': 3
        }
        self.id2word = {
            0: '<bos>', 
            1: '<pad>', 
            2: '<eos>', 
            3: '<unk>'
        }
        idx = 4
        with open(extdict, 'r') as f: 
            lines = f.read().split('\n')
            lines = [x for x in lines if x != '']
        lines = [x.split(' ')[0] for x in lines]
        for i, x in enumerate(lines): 
            self.word2id[x] = i + idx 
            self.id2word[i + idx] = x 
        

    def build_vocab_all_words(self, data): 
        # use all possible words

        # sort all possible words to facilitate adaptive input/softmax
        counts = {}
        for seq in data: 
            for token in seq.split(' ') + ['<eos>']: 
                if token != '': 
                    if token not in counts: 
                        counts[token] = 1
                    else: 
                        counts[token] += 1
                # if token == '': 
                #     print(f"seq is [{seq}]")
                #     exit(0)
        counts = sorted(counts.items(), key=lambda item: item[1])[::-1]

        # special symbols match order of fairseq vocab
        # see: https://github.com/pytorch/fairseq/blob/1479d311d5/fairseq/data/dictionary.py
        self.word2id = {
            '<bos>': 0, 
            '<pad>': 1, 
            '<eos>': 2, 
            '<unk>': 3
        }
        self.id2word = {
            0: '<bos>', 
            1: '<pad>', 
            2: '<eos>', 
            3: '<unk>'
        }
        idx = 4
        for w, c in counts: 
            if w not in self.word2id: 
                self.word2id[w] = idx 
                self.id2word[idx] = w
                idx += 1
    
    def tokenize_data_all_words(self, data): 
        # tokenize all sentences given vocab built from all words
        rst = []
        for seq in data: 
            tmp = []
            for token in seq.split(' ') + ['<eos>']: 
                if token != '': 
                    if token in self.word2id: 
                        idx = self.word2id[token]
                    else: 
                        idx = self.word2id['<unk>']
                    tmp += [ idx ]
            if tmp: 
                rst += [tmp]
        return rst 
    
    def get_size(self): 
        return len(self.word2id)
    
    def get_pad(self): 
        return self.word2id['<pad>']

    def get_stat(self): 
        print(f"vocab size is {self.get_size()}")


def main(): 

    from memxfmr.tools.utils import get_rawdata

    train_sents = get_rawdata('wikitext-103', 'train', '../../')
    valid_sents = get_rawdata('wikitext-103', 'valid', '../../')
    test_sents = get_rawdata('wikitext-103', 'test', '../../')
    print(f"{len(train_sents)} train sentences")
    print(f"{len(valid_sents)} valid sentences")
    print(f"{len(test_sents)} test sentences")
    
    tokenizer = Tokenizer(mode='ext_dict', extdict='../../data/wikitext-103/dict.txt')
    tokenizer.get_stat()

    tokens = tokenizer.tokenize_data(valid_sents[:4])
    for seq in tokens: 
        print()
        print(seq)


    # tokenizer = Tokenizer(data=train_sents+valid_sents+test_sents)
    # tokenizer.get_stat()
    # print(tokenizer.tokenize_data(['this is code', 'for a project']))
    # print(tokenizer.get_pad())


    # print(f"try enwik8")

    # with open("../../data/enwik8/train", 'r') as f: 
    #     train_sents = f.read().split('\n')
    # print(f"{len(train_sents)} sentences")

    # tokenizer = Tokenizer(data=train_sents)
    # tokenizer.get_stat()

    # print(tokenizer.tokenize_data(['101 114 110 32', '65 110 99 105 101 110 116 32']))
    # print(tokenizer.get_pad())
    # print(tokenizer.word2id['<eos>'])



if __name__ == "__main__": main()