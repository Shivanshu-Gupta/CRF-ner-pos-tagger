import ntpath
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

def output(string, filepath=None, silent=False):
    if not silent: print(string)
    if filepath is not None:
        with open(filepath, 'w') as outf:
            outf.write(string + '\n')

def get_search_name(dataset='ner', model='simple', emb=True, enc=1, sep='-', hyperopt=False):
    if hyperopt:
        name = ['hyperopt']
    else:
        name = ['search']

    name.append(dataset)

    if model == 'crf':
        name.append('crf')
    else:
        name.append('simple')

    if emb:
        name.append('emb')

    if enc == 1:
        name.append('enc1')
    elif enc == 2:
        name.append('enc2')

    return sep.join(name)

def get_model_name(model='simple', emb=True, enc=1, sep='-'):
    name = []
    if model == 'crf':
        name.append('crf')
    else:
        name.append('simple')

    if emb:
        name.append('emb')

    if enc != 0:
        name.append('enc')
    return sep.join(name)

def get_configuration_name(emb=True, enc=1, sep='+'):
    name = []
    if emb: name.append('\textsc{pretrained}')
    else: name.append('\textsc{random}')
    if enc != 0: name.append('\textsc{encoder}')
    return sep.join(name)
