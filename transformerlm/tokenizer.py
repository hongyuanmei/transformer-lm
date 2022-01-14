class Tokenizer: 
    "tokenize given input"
    def __init__(self, mode='use_all', data=None): 
        assert data is not None, f"do not support external tokenizer yet"
        self.mode = mode 
        if mode == 'use_all': 
            self.build_vocab = self.build_vocab_all_words
            self.tokenize_data = self.tokenize_data_all_words
        else: 
            raise Exception(f"Unknown mode : {mode}")
        self.build_vocab(data)
    
    def build_vocab_all_words(self, data): 
        # use all possible words
        self.word2id = {'<pad>':0}
        self.id2word = {0:'<pad>'}
        idx = 1
        for seq in data: 
            for token in seq.split(' ') + ['<eos>']: 
                if token not in self.word2id: 
                    self.word2id[token] = idx 
                    self.id2word[idx] = token
                    idx += 1
        if '<unk>' not in self.word2id: 
            self.word2id['<unk>'] = idx 
            self.id2word[idx] = '<unk>'
    
    def tokenize_data_all_words(self, data): 
        # tokenize all sentences given vocab built from all words
        rst = []
        for seq in data: 
            tmp = []
            for token in seq.split(' ') + ['<eos>']: 
                if token in self.word2id: 
                    idx = self.word2id[token]
                else: 
                    idx = self.word2id['<unk>']
                tmp += [ idx ]
            rst += [tmp]
        return rst 
    
    def get_size(self): 
        return len(self.word2id)
    
    def get_pad(self): 
        return self.word2id['<pad>']

    def get_stat(self): 
        print(f"vocab size is {self.get_size()}")


def main(): 

    with open("../../data/wikitext-2/wiki.train.tokens", 'r') as f: 
        train_sents = f.read().split('\n')
    
    print(f"{len(train_sents)} sentences")

    tokenizer = Tokenizer(data=train_sents)
    tokenizer.get_stat()

    print(tokenizer.tokenize_data(['this is code', 'for a project']))

    print(tokenizer.get_pad())


if __name__ == "__main__": main()