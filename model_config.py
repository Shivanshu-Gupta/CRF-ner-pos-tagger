from param import *
from search_analysis import get_best_configs


def get_default_tagger_params(dataset='ner', model='simple'):
    params: TaggingParams = TaggingParams(dataset=dataset)
    if model == 'crf':
        params.model.type = 'neural_crf.NeuralCrf'
    if dataset == 'pos':
        params.training.num_epochs = 40
    params.is_grid = False
    return params


def get_best_tagger_params(dataset='ner', model='simple', emb=False, enc=0):
    best_configs = get_best_configs()
    params: TaggingParams = best_configs[(dataset, model, emb, enc)]
    params.search_name = ''
    params.is_grid = False
    return params
# Dataset   model    emb enc  Dev Accuracy                              embedding     lr
# 0   ner  simple  False   0     95.295348                                     25  0.010
# 1   ner  simple  False   1     95.365046                                     50  0.030
# 2   ner  simple  False   2     95.190800                                     50  0.010
# 3   ner  simple   True   0     95.199512   embeddings/glove.twitter.27B.25d.bin  0.100
# 4   ner  simple   True   1     96.096881  embeddings/glove.twitter.27B.200d.bin  0.003
# 5   ner  simple   True   2     95.913922   embeddings/glove.twitter.27B.25d.bin  0.001
# 6   ner     crf  False   0     95.426032                                     50  0.003
# 7   ner     crf  False   1     95.452169                                     50  0.030
# 8   ner     crf  False   2     95.225649                                     50  0.010
# 9   ner     crf   True   0     95.652553  embeddings/glove.twitter.27B.100d.bin  0.030
# 10  ner     crf   True   1     96.035895  embeddings/glove.twitter.27B.200d.bin  0.010
# 11  pos  simple  False   0     81.163567                                     50  0.100
# 12  pos  simple  False   1     82.927992                                     25  0.100
# 13  pos  simple   True   0     83.309490  embeddings/glove.twitter.27B.100d.bin  0.100
# 14  pos  simple   True   1     86.504530  embeddings/glove.twitter.27B.100d.bin  0.001
# 15  pos     crf  False   0     83.834049                                     50  0.100
# 16  pos     crf  False   1     83.071054                                     50  0.030
# 17  pos     crf   True   0     84.787792   embeddings/glove.twitter.27B.50d.bin  0.100
# 18  pos     crf   True   1     86.552217  embeddings/glove.twitter.27B.200d.bin  0.001

