import attr
import ntpath

def load_embeddings(token_vocab, embedding_dim=None, embedding_path=None):
    import torch
    import torch.nn as nn
    import numpy as np

    # initialize embeddings randomly
    if embedding_path is None:
        embeddings = torch.nn.Embedding(len(token_vocab), embedding_dim)
    # read in pretrained embeddings
    else:
        # word_list = []
        # embeddings_list = []
        # with open(embedding_path, encoding='utf-8') as f:
        #     for line in f:
        #         line = line.split()
        #         word_list.append(line[0])
        #         embeddings_list.append(torch.Tensor(list(map(float, line[1:]))))

        # # create embeddings for special tokens (e.g. UNK)
        # for _ in range(len(token_vocab.special_tokens)):
        #     embeddings_list.append(torch.FloatTensor(embedding_dim).uniform_(-0.1, 0.1))
        # # init a random Embedding object
        # embeddings = torch.nn.Embedding(len(embeddings_list), embedding_dim)
        # # set embedding weights to the embeddings we loaded
        # embeddings.weight.data.copy_(torch.vstack(embeddings_list))

        filename = ntpath.basename(embedding_path)
        if filename.startswith("word2vec") or filename.startswith("glove"):
            from gensim.models.keyedvectors import KeyedVectors
            emb_model = KeyedVectors.load_word2vec_format(embedding_path, binary=True)
        elif filename.startswith("fasttext"):
            from gensim.models.fasttext import load_facebook_vectors
            emb_model = load_facebook_vectors('fasttext_twitter_raw.bin')
        else:
            raise Exception("Unsuported word vectors")

        word_list = list(emb_model.key_to_index.keys())
        embedding_dim = emb_model.vector_size
        weights = torch.vstack([torch.FloatTensor(emb_model.vectors),
                                *(torch.FloatTensor(embedding_dim).uniform_(-0.1, 0.1)
                                  for _ in token_vocab.special_tokens)])
        # init a random Embedding object
        embeddings = torch.nn.Embedding(weights.shape[0], embedding_dim)
        # set embedding weights to the embeddings we loaded
        embeddings.weight.data.copy_(weights)

        # update word list in token vocabulary with words from embedding file
        token_vocab.word_list = word_list

    return embeddings


def create_object_from_class_string(module_name, class_name, parameters):
    import importlib
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    instance = class_(**parameters)
    return instance


def load_object_from_dict(parameters, **kwargs):
    if not isinstance(parameters, dict):
        parameters = attr.asdict(parameters)
    parameters.update(kwargs)
    type = parameters.pop('type')
    if type is None:
        return None
    else:
        type = type.split('.')
        module_name, class_name = '.'.join(type[:-1]), type[-1]
        return create_object_from_class_string(module_name, class_name, parameters)

# def load_object_from_dict(parameters: dict, **kwargs):
#     parameters.update(kwargs)
#     clazz = parameters.pop('type')
#     if clazz is None:
#         return None
#     else:
#         return clazz(**parameters)
