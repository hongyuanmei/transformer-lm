import torch
import torch.nn as nn
import torch.nn.functional as F

import functools
import operator

# learn from: 
# fairseq: https://github.com/pytorch/fairseq/blob/d421749323/fairseq/modules/adaptive_softmax.py
# pytorch: https://pytorch.org/docs/stable/_modules/torch/nn/modules/adaptive.html#AdaptiveLogSoftmaxWithLoss
# use fairseq init to facilitate parameter sharing with adaptive input
# use pytorch forward since it is more efficient (50% memory reduction) and works well with DDP

class TiedLinear(nn.Module):
    def __init__(self, weight, transpose):
        super().__init__()
        self.weight = weight
        self.transpose = transpose

    def forward(self, input):
        return F.linear(input, self.weight.t() if self.transpose else self.weight)


class TiedHeadModule(nn.Module):
    def __init__(self, weights, input_dim, num_classes):
        super().__init__()
        tied_emb, _ = weights
        self.num_words, emb_dim = tied_emb.size()

        self.word_proj = TiedLinear(tied_emb, transpose=False)
        if input_dim != emb_dim:
            self.word_proj = nn.Sequential(
                nn.Linear(input_dim, emb_dim, bias=False),
                self.word_proj,
            )

        self.class_proj = nn.Linear(input_dim, num_classes, bias=False)
        self.out_dim = self.num_words + num_classes

        self.register_buffer("_float_tensor", torch.FloatTensor(1))

    def forward(self, input):
        inp_sz = functools.reduce(operator.mul, input.shape[:-1], 1)
        out = self._float_tensor.new(inp_sz, self.out_dim)
        out[:, : self.num_words] = self.word_proj(input.view(inp_sz, -1))
        out[:, self.num_words :] = self.class_proj(input.view(inp_sz, -1))
        return out


class AdaptiveSoftmax(nn.Module):
    """
    This is an implementation of the efficient softmax approximation for
    graphical processing units (GPU), described in the paper "Efficient softmax
    approximation for GPUs" (http://arxiv.org/abs/1609.04309).
    """

    def __init__(
        self,
        vocab_size,
        input_dim,
        cutoff,
        dropout,
        factor=4.0,
        adaptive_inputs=None,
        tie_proj=False,
    ):
        super().__init__()

        if vocab_size > cutoff[-1]:
            cutoff = cutoff + [vocab_size]
        else:
            assert (
                vocab_size == cutoff[-1]
            ), "cannot specify cutoff larger than vocab size"

        output_dim = cutoff[0] + len(cutoff) - 1

        self.vocab_size = vocab_size
        self.cutoff = cutoff
        self.dropout_module = nn.Dropout(dropout)
        self.input_dim = input_dim
        self.factor = factor

        self.lsm = nn.LogSoftmax(dim=1)

        if adaptive_inputs is not None:
            self.head = TiedHeadModule(
                adaptive_inputs.weights_for_band(0),
                input_dim,
                len(cutoff) - 1,
            )
        else:
            self.head = nn.Linear(input_dim, output_dim, bias=False)
        
        self._make_tail(adaptive_inputs, tie_proj)

        def init_weights(m):
            if (
                hasattr(m, "weight")
                and not isinstance(m, TiedLinear)
                and not isinstance(m, TiedHeadModule)
            ):
                nn.init.xavier_uniform_(m.weight)

        self.apply(init_weights)

        # above is all from fairseq 
        # below is from pytorch (except register_buffer)

        self.cutoffs = self.cutoff #+ [self.vocab_size]
        self.shortlist_size = self.cutoff[0]

        self.register_buffer("version", torch.LongTensor([1]))

    def _make_tail(self, adaptive_inputs=None, tie_proj=False):
        self.tail = nn.ModuleList()
        for i in range(len(self.cutoff) - 1):
            dim = int(self.input_dim // self.factor ** (i + 1))

            tied_emb, tied_proj = (
                adaptive_inputs.weights_for_band(i + 1)
                if adaptive_inputs is not None
                else (None, None)
            )

            if tied_proj is not None:
                if tie_proj:
                    proj = TiedLinear(tied_proj, transpose=True)
                else:
                    proj = nn.Linear(tied_proj.size(0), tied_proj.size(1), bias=False)
            else:
                proj = nn.Linear(self.input_dim, dim, bias=False)

            if tied_emb is None:
                out_proj = nn.Linear(
                    dim, self.cutoff[i + 1] - self.cutoff[i], bias=False
                )
            else:
                out_proj = TiedLinear(tied_emb, transpose=False)

            m = nn.Sequential(
                proj,
                nn.Dropout(self.dropout_module.p),
                out_proj,
            )

            self.tail.append(m)

    def upgrade_state_dict_named(self, state_dict, name):
        version_name = name + ".version"
        if version_name not in state_dict:
            raise Exception("This version of the model is no longer supported")

    
    def forward(self, input, target): 
        
        if input.size(0) != target.size(0):
            raise RuntimeError('Input and target should have the same size '
                               'in the batch dimension.')

        used_rows = 0
        batch_size = target.size(0)

        output = input.new_zeros(batch_size)
        gather_inds = target.new_empty(batch_size)

        cutoff_values = [0] + self.cutoffs

        # hack : to use all params 
        # to use pytorch DDP, we have to use all params of model 
        # to not affect the actual result, we choose to access all params 
        # and then accumulate them and multiply by 0.0
        tmp = 0.0 
        tmpind = torch.zeros((self.input_dim,), dtype=torch.float32).cuda()

        for i in range(len(cutoff_values) - 1):

            if i == 0: 
                # access head 
                tmp += self.head(tmpind).sum()
            else: 
                # access tail 
                tmp += self.tail[i-1](tmpind).sum()

            low_idx = cutoff_values[i]
            high_idx = cutoff_values[i + 1]

            target_mask = (target >= low_idx) & (target < high_idx)
            row_indices = target_mask.nonzero().squeeze()

            if row_indices.numel() == 0:
                continue

            if i == 0:
                gather_inds.index_copy_(0, row_indices, target[target_mask])

            else:
                relative_target = target[target_mask] - low_idx
                input_subset = input.index_select(0, row_indices)

                cluster_output = self.tail[i - 1](input_subset)
                cluster_index = self.shortlist_size + i - 1

                gather_inds.index_fill_(0, row_indices, cluster_index)

                cluster_logprob = F.log_softmax(cluster_output, dim=1)
                local_logprob = cluster_logprob.gather(1, relative_target.unsqueeze(1))
                output.index_copy_(0, row_indices, local_logprob.squeeze(1))

            used_rows += row_indices.numel()

        if used_rows != batch_size:
            raise RuntimeError("Target values should be in [0, {}], "
                               "but values in range [{}, {}] "
                               "were found. ".format(self.vocab_size - 1,
                                                     target.min().item(),
                                                     target.max().item()))

        head_output = self.head(input)
        head_logprob = F.log_softmax(head_output, dim=1)
        output += head_logprob.gather(1, gather_inds.unsqueeze(1)).squeeze()
        loss = (-output).mean()
        
        loss += tmp * 0.0 

        return output, loss


def main(): 

    import gc
    from adainput import AdaptiveInput
    import random

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()

    print(f"test adaptive softmax")

    vocab_size = 267744
    padding_idx = 0
    dim = 1024
    T = 3072
    factor = 4
    cutoff = [20000, 60000]

    torch.manual_seed(0)
    random.seed(0)

    adainput = AdaptiveInput(
        vocab_size = vocab_size,
        padding_idx = padding_idx,
        initial_dim = dim,
        factor = factor,
        output_dim = dim,
        cutoff = cutoff,
    ).cuda()

    torch.manual_seed(0)
    random.seed(0)

    adasoftmax = AdaptiveSoftmax(
        vocab_size = vocab_size,
        input_dim = dim,
        cutoff = cutoff,
        dropout = 0.1,
        factor = factor,
        adaptive_inputs = adainput,
        tie_proj=True,
    ).cuda()

    print("\nadaptive softmax")
    for k, v in adasoftmax.state_dict().items(): 
        print(f"{k}  :  {v.size()}")

    torch.manual_seed(0)
    random.seed(0)

    x = torch.randint(0, 267744, (1, T,), dtype=torch.int64).cuda()
    h = adainput(x)
    print(f"\nh")
    print(f"size is : {h.size()}")

    h = h.contiguous().view(-1, dim)
    x = x.contiguous().view(-1)

    outs, loss = adasoftmax(h, x)
    
    print("\nlogprob")
    print(f"logprob size : {outs.size()}")

    print("\nshow some logprobs")
    print(outs[:10])

    print("\nloss")
    print(loss)

    # track memory
    torch.cuda.synchronize()
    print("\nMax memory used by tensors")
    xxx = torch.cuda.max_memory_allocated()
    ggg = xxx//1000000000
    mmm = (xxx-ggg*1000000000) // 1000000
    print(f"{xxx}B")
    print(f"{ggg}GB-{mmm} MB")

if __name__ == "__main__": main()