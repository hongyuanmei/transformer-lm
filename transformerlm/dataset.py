import numpy as np 
import torch 
from torch.utils.data import Dataset
from utils import subsequent_mask

"""
important: concat all data and org them to batches
learned from: https://github.com/pytorch/examples/blob/master/word_language_model/main.py
"""

class Dataset(Dataset): 

    def __init__(self, data, batch_size, bptt, pad_idx): 
        concat_data = [x for sub in data for x in sub] # [[...], [...]] -> [..., ...]
        concat_data = torch.tensor(concat_data).type(torch.int64)
        # Work out how cleanly we can divide the dataset into bsz parts.
        nbatch = concat_data.size(0) // batch_size
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        concat_data = concat_data.narrow(0, 0, nbatch * batch_size)
        # Evenly divide the data across the bsz batches.
        self.data = concat_data.view(batch_size, -1).t()
        # Starting from sequential data, batchify arranges the dataset into columns.
        # For instance, with the alphabet as the sequence and batch size 4, we'd get
        # ┌ a g m s ┐
        # │ b h n t │
        # │ c i o u │
        # │ d j p v │
        # │ e k q w │
        # └ f l r x ┘.
        # These columns are treated as independent by the model, which means that the
        # dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
        # batch processing.
        self.batch_size = batch_size
        self.bptt = bptt
        self.pad_idx = pad_idx

    
    def __len__(self): 
        return self.data.size(0)
    
    def __getitem__(self, idx): 
        raise NotImplementedError
    
    def get_batch(self, i): 
        seq_len = min(self.bptt, self.__len__() - 1 - i)
        data = self.data[i:i+seq_len].t().contiguous()
        target = self.data[i+1:i+1+seq_len].t().contiguous()
        attn_mask = self.make_attn_mask(data, self.pad_idx)
        target_mask = self.make_target_mask(target, self.pad_idx)
        return data, target, attn_mask, target_mask

    
    def make_attn_mask(self, sent, pad_idx): 
        "create a mask to hide padding and future words"
        mask = (sent != pad_idx).unsqueeze(-2)
        mask = mask & subsequent_mask(sent.size(-1)).type_as(mask)
        return mask
    
    def make_target_mask(self, sent, pad_idx): 
        return sent != pad_idx


def main(): 

    data = [
        [1, 2, 3], 
        [1, 2], 
        [1, 2, 3, 4], 
        [1, 2], 
        [1, 2, 3, 4, 5], 
    ]

    bsz = 4
    bptt = 3
    pad_idx = 0

    dataset = Dataset(data, bsz, bptt, pad_idx)
    # dataset.local_sort(4)
    # dataset.local_sort(2)
    print(f"\nall data")
    print(dataset.data)
    # ┌ 1 2 4 2 ┐
    # │ 2 1 1 3 │
    # │ 3 2 2 4 │
    # └ 1 3 1 5 ┘

    print(f"\npossible range")
    print(range(0, len(dataset) - 1, bptt))

    i = 0

    data, target, attn_mask, target_mask = dataset.get_batch(i)
    
    print(f"\nbatch data")
    print(data)
    print(target)
    print(attn_mask)
    print(target_mask)
    print(torch.sum(target_mask))


if __name__ == "__main__": main()