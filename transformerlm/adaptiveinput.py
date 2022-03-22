from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

# learn from: https://github.com/pytorch/fairseq/blob/d421749323/fairseq/modules/adaptive_input.py
# simplification: no quantization
# adaptive input assume the vocabulary is already sorted

class AdaptiveInput(nn.Module): 

    def __init__(self, 
        vocab_size: int,
        padding_idx: int,
        initial_dim: int,
        factor: float,
        output_dim: int,
        cutoff: List[int],
    ):
        super(AdaptiveInput, self).__init__()

        if vocab_size > cutoff[-1]:
            cutoff = cutoff + [vocab_size]
        else:
            assert (
                vocab_size == cutoff[-1]
            ), "cannot specify cutoff larger than vocab size"

        self.cutoff = cutoff
        self.embedding_dim = output_dim
        self.padding_idx = padding_idx

        self.embeddings = nn.ModuleList()
        for i in range(len(self.cutoff)):
            prev = self.cutoff[i - 1] if i > 0 else 0
            size = self.cutoff[i] - prev
            dim = int(initial_dim // (factor**i))
            seq = nn.Sequential(
                nn.Embedding(size, dim, self.padding_idx),
                nn.Linear(dim, output_dim, bias=False)
            )
            self.embeddings.append(seq)
            self.padding_idx = None
        self.padding_idx = padding_idx

        def init_weights(m):
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=m.weight.shape[1] ** -0.5)
                nn.init.constant_(m.weight[padding_idx], 0)
            elif hasattr(m, "weight"):
                nn.init.xavier_uniform_(m.weight)

        self.apply(init_weights)

        self.register_buffer("_float_tensor", torch.FloatTensor(1))

    def weights_for_band(self, band: int):
        return self.embeddings[band][0].weight, self.embeddings[band][1].weight

    def forward(self, input: torch.Tensor):
        result = self._float_tensor.new(input.shape + (self.embedding_dim,))

        # hack : to use all params 
        # to use pytorch DDP, we have to use all params of model 
        # to not affect the actual result, we choose to access all params 
        # and then accumulate them and multiply by 0.0
        tmp = 0.0 
        tmpind = torch.zeros((1,), dtype=torch.int64).cuda()

        for i in range(len(self.cutoff)):
            mask = input.lt(self.cutoff[i])
            if i > 0:
                mask.mul_(input.ge(self.cutoff[i - 1]))
                chunk_input = input[mask] - self.cutoff[i - 1]
            else:
                chunk_input = input[mask]
            if mask.any():
                result[mask] = self.embeddings[i](chunk_input)
            
            tmp += self.embeddings[i](tmpind).sum()
        
        result += tmp * 0.0

        return result


def main(): 

    print(f"test adaptive input")

    ada = AdaptiveInput(
        vocab_size = 267744,
        padding_idx = 0,
        initial_dim = 1024,
        factor = 4,
        output_dim = 1024,
        cutoff = [20000, 60000],
    )

    for k, v in ada.state_dict().items(): 
        print(f"{k}  :  {v.size()}")

    x = torch.randint(0, 267744, (8, 512), dtype=torch.int64)

    print(f"\nada(x) has size = {ada(x).size()}")

if __name__ == "__main__": main()