from copy import deepcopy

from gensim.utils import simple_preprocess

from param import *

best_embedding_param = 'embeddings/glove.twitter.27B.25d.bin'
# best_embedding_param = 'embeddings/glove.twitter.27B.200d.bin'
best_encoder = EncoderParams(type='torch.nn.LSTM')

def tagger(dataset='ner', model='simple'):
    params = TaggingParams(dataset=dataset)
    if model == 'crf':
        params.model.type = 'neural_crf.NeuralCrf'
    if dataset == 'ner' and model == 'simple':
        # params.optimizer = SGDOptimizerParams(type='torch.optim.SGD',
        #                                       lr=0.05721092247217258,
        #                                       momentum=0.6872224143884547)
        params.optimizer = OptimizerParams(lr=0.0029397976202716874)
    return params

def tagger_w_emb(dataset='ner', model='simple'):
    params = TaggingParams(dataset=dataset)
    if model == 'crf':
        params.model.type = 'neural_crf.NeuralCrf'
    params.model.embedding_param = best_embedding_param
    return params

def tagger_w_enc(dataset='ner', model='simple'):
    params = TaggingParams(dataset=dataset)
    if model == 'crf':
        params.model.type = 'neural_crf.NeuralCrf'
    params.model.embedding_param = ''
    params.model.encoder = deepcopy(best_encoder)
    return params

def tagger_w_enc_emb(dataset='ner', model='simple'):
    params = TaggingParams(dataset=dataset)
    if model == 'crf':
        params.model.type = 'neural_crf.NeuralCrf'
    params.model.embedding_param = best_embedding_param
    params.model.encoder = deepcopy(best_encoder)
    return params


# ner/pos
# - simple
#     - base
#     - pretrained emb
#     - encoder
#     - encoder + embedding
# - crf
#     - base
#     - pretrained emb
#     - encoder
#     - encoder + embedding
