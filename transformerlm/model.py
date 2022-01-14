import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from utils import clones, attention

# in-house implementation of Transformer
# largely reply on Susha Rush annotated transformer: https://nlp.seas.harvard.edu/2018/04/03/attention.html
# but this model is a generative model 

class TransformerLM(nn.Module): 

    def __init__(self, vocab_size, 
        n_layers, d_emb, d_model, n_heads, d_ffn, d_softmax, tie_emb, dropout):
        super(TransformerLM, self).__init__()
        # hyper-params
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.d_emb = d_emb
        self.d_model = d_model 
        self.n_heads = n_heads
        self.d_ffn = d_ffn
        self.d_softmax = d_softmax
        self.tie_emb = tie_emb
        self.dropout = dropout

        self.embed = Embeddings(vocab_size, d_emb)
        # d_ff = d_model is our simplification
        if d_emb != d_model: # word emb dim not match model dim 
            self.input_proj = nn.Linear(d_emb, d_model, bias=False)
        self.position = PositionEmbedding(d_model, dropout)
        self.decoder = Decoder(n_layers, d_model, n_heads, d_ffn, dropout)
        if d_model != d_softmax: # output layer emb dim not match model dim
            self.output_proj = nn.Linear(d_model, d_softmax, bias=False)
        self.generator = Generator(d_softmax, vocab_size)

        if self.tie_emb == 1: 
            # tie input and out embeddings
            # if we use adaptive softmax, we also have to use adaptive input
            assert d_emb == d_softmax, f"dimension mismatch: d_emb={d_emb} d_softmax={d_softmax}"
            self.generator.tie(self.embed.get_weight())
        elif self.tie_emb == 0: 
            pass # do nothing 
        else: 
            raise Exception(f"Unknown tie_emb : {self.tie_emb}")

        # according to Sasha, this is important 
        for p in self.parameters(): 
            if p.dim() > 1: 
                nn.init.xavier_uniform(p)

    def forward(self, sent, mask, output='prob'):
        x = self.embed(sent)
        if self.d_emb != self.d_model: 
            x = self.input_proj(x)
        hidden_state = self.decoder(self.position(x), mask)
        if output == 'hidden': 
            return hidden_state
        elif output == 'prob': 
            if self.d_model != self.d_softmax: 
                hidden_state = self.output_proj(hidden_state)
            return self.generator(hidden_state)
        else: 
            raise Exception(f"Unknown output type : {output}")

class Generator(nn.Module): 

    def __init__(self, d_model, vocab_size): 
        super(Generator, self).__init__()
        self.linear = nn.Linear(d_model, vocab_size)
    
    def forward(self, x): 
        return F.log_softmax(self.linear(x), dim=-1)
    
    def tie(self, weight): 
        #print(weight.shape)
        #print(self.linear.weight.shape)
        self.linear.weight = weight


class SublayerConnection(nn.Module): 
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, d_model, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class Decoder(nn.Module): 

    def __init__(self, n_layers, d_model, n_heads, d_ffn, dropout): 
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(
            [ DecoderLayer(d_model, n_heads, d_ffn, dropout) for _ in range(n_layers) ])
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x, mask): 
        for layer in self.layers: 
            x = layer(x, mask)
        return self.norm(x)


class DecoderLayer(nn.Module): 

    def __init__(self, d_model, n_heads, d_ffn, dropout): 
        super(DecoderLayer, self).__init__()
        self.attn = MultiHeadedAttention(d_model, n_heads, dropout)
        self.ffn = PositionWiseFeedForward(d_model, d_ffn, dropout)
        self.sublayers = nn.ModuleList(
            [ SublayerConnection(d_model, dropout) for _ in range(2) ])
    
    def forward(self, x, mask): 
        x = self.sublayers[0](x, lambda x: self.attn(x, x, x, mask))
        return self.sublayers[1](x, self.ffn)


class MultiHeadedAttention(nn.Module): 

    def __init__(self, d_model, n_heads, dropout): 
        super(MultiHeadedAttention, self).__init__()
        assert d_model % n_heads == 0, "dim(model) must be a multiple of num(head)"
        self.d_key = d_model // n_heads
        self.n_heads = n_heads
        self.linears = nn.ModuleList(
            [ nn.Linear(d_model, d_model) for _ in range(4) ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None: 
            # same mask applied to all h heads 
            mask = mask.unsqueeze(1)
        batch_size = query.size(0)

        # do linear proj in batch from d_model -> h x d_key 
        query, key, value = [
            l(x).view(batch_size, -1, self.n_heads, self.d_key).transpose(1,2) \
            for l, x in zip(self.linears, (query, key, value))
        ]

        # apply attention on all the proj vectors in batch 
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout )
        
        # concat and linear proj
        x = x.transpose(1, 2).contiguous().view(
            batch_size, -1, self.n_heads * self.d_key )

        return self.linears[-1](x)

    
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
        return self.lut(x) # * math.sqrt(self.d_model)
    
    def get_weight(self): 
        return self.lut.weight


class PositionEmbedding(nn.Module): 

    def __init__(self, d_model, dropout, max_len=5000): 
        super(PositionEmbedding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        # compute the positional embeddings once in log space 
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * \
            -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x): 
        # x size : B x T x d_model
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)