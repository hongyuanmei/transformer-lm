# transformer-lm
a (nearly) minimal pytorch implementation for transformer language model

this repo is only for education use

model code largely relies on [annotated transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)

data processing is adapted from [pytorch word-language-model](https://github.com/pytorch/examples/tree/master/word_language_model)

many design choices (e.g., emb proj layers) are learned from [Baevski and Auli 2019](https://arxiv.org/abs/1809.10853)

it achieves valid ppl = 25 on Wikitext-103
