import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.utils.checkpoint
import math

from utils import clones, attention
from adainput import AdaptiveInput
from adasoftmax import AdaptiveSoftmax

# in-house implementation of Transformer
# largely reply on Susha Rush annotated transformer: https://nlp.seas.harvard.edu/2018/04/03/attention.html
# but this model is a generative model 

# also consult fairseq source code
# since that code achieves SOTA on wikitext and enwik
# https://github.com/pytorch/fairseq/blob/d421749323/fairseq/modules/transformer_layer.py
# eps. adaptive input & softmax, order of operation, ...
# also uses gradient checkpointing

class TransformerLM(nn.Module): 

    def __init__(self, 
        vocab_size, n_layers, d_model, n_heads, d_ffn, 
        ada, adacutoff, pad_idx, tie_emb, dropout):
        super(TransformerLM, self).__init__()
        # hyper-params
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.d_model = d_model 
        self.n_heads = n_heads
        self.d_ffn = d_ffn
        self.scaling = math.sqrt(d_model)
        self.ada = ada 
        self.adacutoff = sorted(adacutoff)
        self.pad_idx = pad_idx
        self.tie_emb = tie_emb
        self.dropout = dropout

        # input embeddings : adaptive or not 
        if ada: 
            self.embed = AdaptiveInput(
                vocab_size, pad_idx, d_model, 4.0, d_model, self.adacutoff)
        else: 
            self.embed = Embeddings(vocab_size, d_model)
        # positional encoding
        self.position = PositionEmbedding(d_model, dropout)
        # decoder : Transformer layers
        self.decoder = Decoder(n_layers, d_model, n_heads, d_ffn, dropout)
        # softmax : adaptive or not 
        if ada: 
            tie_proj = True if tie_emb else False 
            self.generator = AdaptiveSoftmax(
                vocab_size, d_model, self.adacutoff, dropout, 4.0, self.embed, tie_proj)
        else: 
            self.generator = Generator(d_model, vocab_size)
            if tie_emb: 
                self.generator.tie(self.embed.get_weight())
        
        # according to Sasha, this is important 
        for p in self.parameters(): 
            if p.dim() > 1: 
                nn.init.xavier_uniform(p)

    def forward(self, sent, mask):
        x = self.embed(sent)
        x *= self.scaling # fairseq hack
        x = self.position(x)
        hidden_state = self.decoder(x, mask)
        return hidden_state

class Generator(nn.Module): 

    def __init__(self, d_model, vocab_size): 
        super(Generator, self).__init__()
        self.linear = nn.Linear(d_model, vocab_size)
    
    def forward(self, x): 
        return F.log_softmax(self.linear(x), dim=-1)
    
    def tie(self, weight): 
        self.linear.weight = weight


class Decoder(nn.Module): 

    def __init__(self, n_layers, d_model, n_heads, d_ffn, dropout): 
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(
            [ DecoderLayer(d_model, n_heads, d_ffn, dropout) for _ in range(n_layers) ])

    def forward(self, x, mask): 
        for layer in self.layers: 
            # gradient checkpointing
            def create_custom_forward(module): 
                def custom_forward(y): 
                    return module(y, mask)
                return custom_forward
            # gradient checkpoing trick to save GPU memory
            # https://medium.com/tensorflow/fitting-larger-networks-into-memory-583e3c758ff9

            x = torch.utils.checkpoint.checkpoint(
                create_custom_forward(layer), x)
            #x = layer(x, mask)
        return x 


class DecoderLayer(nn.Module): 
    # we consult fairseq for tech details
    # https://github.com/pytorch/fairseq/blob/d421749323/fairseq/modules/transformer_layer.py

    def __init__(self, d_model, n_heads, d_ffn, dropout): 
        super(DecoderLayer, self).__init__()
        self.attn = MultiHeadedAttention(d_model, n_heads, dropout)
        self.ffn = PositionWiseFeedForward(d_model, d_ffn, dropout)
        self.attn_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask): 
        # attn block 
        residual = x 
        x = self.attn_norm(x)
        x = self.attn(x, x, x, mask)
        x = self.dropout(x)
        x = x + residual
        # ffn block 
        residual = x 
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = x + residual
        return x 


class MultiHeadedAttention(nn.Module): 

    def __init__(self, d_model, n_heads, dropout): 
        super(MultiHeadedAttention, self).__init__()
        assert d_model % n_heads == 0, "dim(model) must be a multiple of num(head)"
        self.d_model = d_model
        self.d_key = d_model // n_heads
        self.n_heads = n_heads
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None): 
        """
        adapted from fairseq 
        https://github.com/pytorch/fairseq/blob/main/fairseq/modules/multihead_attention.py#L308-L330
        """
        assert mask is not None 
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)
        attn_mask = ~mask[0,:,:]
        # multi_head_atten_forward uses seq len as dim-0 and batch size as dim-1
        attn_out, _ = F.multi_head_attention_forward(
            query, 
            key, 
            value, 
            self.d_model, 
            self.n_heads, 
            torch.empty([0]).cuda(), 
            torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)), 
            None, # self.bias_k, 
            None, # self.bias_v, 
            False, # self.add_zero_attn
            self.dropout.p, # should be 0.1
            self.out_proj.weight, 
            self.out_proj.bias, 
            self.training, # self.training or self.dropout_module.apply_during_inference 
            None, # key_padding_mask 
            False, # need_weights 
            attn_mask, # attn_mask
            use_separate_proj_weight=True,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
        )
        attn_out = attn_out.transpose(0, 1)
        return attn_out
    
class PositionWiseFeedForward(nn.Module): 
    
    def __init__(self, d_model, d_ff, dropout=0.1): 
        super(PositionWiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x): 
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module): 

    def __init__(self, vocab_size, d_model): 
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
    
    def forward(self, x): 
        # NOTE: not sure if we really need to scale by sqrt(d_model)
        return self.lut(x) * math.sqrt(self.d_model)
    
    def get_weight(self): 
        return self.lut.weight


class PositionEmbedding(nn.Module): 
    """
    fairseq position embedding
    https://github.com/pytorch/fairseq/blob/b554f5ec9067c2f88bf17a736ddb57ea1cab14cc/fairseq/modules/sinusoidal_positional_embedding.py#L15
    """
    def __init__(self, d_model, dropout, max_len=5000): 
        super(PositionEmbedding, self).__init__()
        # dropout is never used
        # we keep it in arg to match interface
        self.dropout = nn.Dropout(dropout)
        self.embedding_dim = d_model
        self.max_len = max_len
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = self.embedding_dim // 2 
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        """
        +2 is fairseq hack 
        https://github.com/pytorch/fairseq/issues/1177
        """
        emb = torch.arange(self.max_len + 2, dtype=torch.float).unsqueeze(
            1
        ) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(
            self.max_len + 2, -1
        )
        if self.embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(self.max_len + 2, 1)], dim=1)
        """
        precompute positions 
        https://github.com/pytorch/fairseq/blob/b554f5ec9067c2f88bf17a736ddb57ea1cab14cc/fairseq/utils.py#L256
        """
        """
        +2 is fairseq hack 
        https://github.com/pytorch/fairseq/issues/1177
        we adopt same design to facilitate using their saved model
        """
        position = torch.arange(0, self.max_len) + 2
        pe = emb.index_select(0, position)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x): 
        # x size : B x T x d_model
        x = x + self.pe[:, :x.size(1)]
        return x

def main(): 

    pe = PositionEmbedding(1024, 0.1)
    print(pe.pe[0,:10,:4])
    print()
    print(pe.pe[0,:10,-5:])

    # check fairseq's position embeddings
    pefs = torch.load('../../tryfairseqwk103/reformat/pe_weights.pt')
    d = pe.pe[0,:3072,:] - pefs[2:,:]
    print(f"total difference^2 = {float((d**2).sum())}")

if __name__ == '__main__': main()