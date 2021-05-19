import attr
import ntpath
from numpy.lib.arraysetops import isin
import torch

def load_embeddings(token_vocab, embedding_param='50'):
    import torch
    glove = [f'embeddings/glove.twitter.27B.{dim}d.bin'
             for dim in [25, 50, 100, 200]]
    word2vec = ['embeddings/word2vec_twitter_tokens.bin']
    fasttext = ['embeddings/fasttext_twitter_raw.bin']
    valid_embedding_paths = set(glove + word2vec + fasttext)

    # read in pretrained embeddings
    if embedding_param in valid_embedding_paths:
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
        embedding_path = embedding_param
        filename = ntpath.basename(embedding_path)
        if filename.startswith("word2vec") or filename.startswith("glove"):
            from gensim.models.keyedvectors import KeyedVectors
            emb_model = KeyedVectors.load_word2vec_format(embedding_path, binary=True, unicode_errors='ignore')
        elif filename.startswith("fasttext"):
            from gensim.models.fasttext import load_facebook_vectors
            emb_model = load_facebook_vectors(embedding_path)
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
    else:
        # initialize embeddings randomly
        embeddings = torch.nn.Embedding(len(token_vocab), int(embedding_param))

    return embeddings


def create_object_from_class_string(module_name, class_name, parameters):
    import importlib
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    instance = class_(**parameters)
    return instance


def load_object_from_dict(parameters, **kwargs):
    if not isinstance(parameters, dict):
        # parameters = attr.asdict(parameters)
        parameters = vars(parameters).copy()
    parameters.update(kwargs)
    type = parameters.pop('type')
    if not type:
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

def get_device(gpu_idx):
    device = 'cpu'
    if gpu_idx != -1 and torch.cuda.is_available():
        device = f'cuda:{gpu_idx}'
    return device

def nest_dict(flat, sep='.'):
    result = {}
    for k, v in flat.items():
        _nest_dict_rec(k, v, result, sep=sep)
    return result

def _nest_dict_rec(k, v, out, sep='.'):
    k, *rest = k.split(sep, 1)
    if rest:
        _nest_dict_rec(rest[0], v, out.setdefault(k, {}))
    else:
        out[k] = v

def output(string, filepath=None):
    print(string)
    if filepath is not None:
        with open(filepath, 'w') as outf:
            outf.write(string + '\n')
