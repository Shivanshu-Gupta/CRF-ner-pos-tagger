import os
import json
import attr
from typing import Union, Optional

from parameters import get_default, Parameters

datasets = ['ner', 'pos']
model_types = ['simple', 'crf']
glove = [f'glove.twitter.27B.{dim}d.bin' for dim in [25, 50, 100, 200]]
word2vec = ['word2vec_twitter_tokens.bin']
fasttext = ['fasttext_twitter_raw.bin']
valid_embeddings = glove

paths = json.load(open('paths.json'))
data_dir = paths['data_dir']
embeddings_dir = paths['embeddings_dir']
log_dir = paths['log_dir']
models_dir = paths['models_dir']
ray_dir = paths['ray_dir']


def disambiguate(o, t):
    lambdas = {
        Union[AdamOptimizerParams, SGDOptimizerParams]: lambda o, _: SGDOptimizerParams if 'momentum' in o else AdamOptimizerParams,
        Union[int, str]: lambda *_: None
    }
    if t in lambdas:
        return lambdas[t](o, t)
    else:
        raise TypeError("Unknown Type")

@attr.s(auto_attribs=True)
class DataParams(Parameters):
    data_path: str = os.path.join(data_dir, "twitter_train.ner")
    max_length: int = 30

@attr.s(auto_attribs=True)
class EncoderParams(Parameters):
    type: str = 'torch.nn.LSTM'
    hidden_size: int = 100
    num_layers: int = 1
    bias: bool = True
    dropout: float = 0
    bidirectional: bool = True


@attr.s(auto_attribs=True)
class ProjectionParams(Parameters):
    type: str = 'torch.nn.Linear'
    out_features: int = 22


@attr.s(auto_attribs=True)
class ModelParams(Parameters):
    type: str = 'models.simple_tagger.SimpleTagger'
    embedding_param: Union[int, str] = 50
    encoder: Optional[EncoderParams] = None
    tag_projection: ProjectionParams = get_default(lambda: ProjectionParams())

    @classmethod
    def get_disambiguators(cls):
        return {Union[int, str]: disambiguate}


@attr.s(auto_attribs=True)
class AdamOptimizerParams(Parameters):
    type: str = 'torch.optim.Adam'
    lr: float = 0.001


@attr.s(auto_attribs=True)
class SGDOptimizerParams(Parameters):
    type: str = 'torch.optim.SGD'
    lr: float = 0.001
    momentum: float = 0.1


@attr.s(auto_attribs=True)
class TrainingParams(Parameters):
    num_epochs: int = 20
    optimizer: Union[AdamOptimizerParams,
                     SGDOptimizerParams] = get_default(lambda: AdamOptimizerParams())

    @classmethod
    def get_disambiguators(cls):
        return {Union[AdamOptimizerParams, SGDOptimizerParams]: disambiguate}


@attr.s(auto_attribs=True)
class TaggingParams(Parameters):
    random_seed: int = 42
    gpu_idx: int = -1
    dataset: str = 'ner'
    train_dataset: DataParams = get_default(lambda: DataParams(data_path=os.path.join(data_dir, "twitter_train.ner")))
    validation_dataset: DataParams = get_default(lambda: DataParams(data_path=os.path.join(data_dir, "twitter_dev.ner")))
    train_batch_size: int = 32
    val_batch_size: int = 64
    model: ModelParams = get_default(lambda: ModelParams())
    training: TrainingParams = get_default(lambda: TrainingParams())
    search_name: str = ''

    def __attrs_post_init__(self):
        if self.dataset == 'pos':
            self.train_dataset.data_path = os.path.join(data_dir, "twitter_train.pos")
            self.validation_dataset.data_path = os.path.join(data_dir, "twitter_dev.pos")
            self.model.tag_projection.out_features = 13
            self.training.num_epochs = 10

# params = TaggingParams()
# params.model.encoder = [EncoderParams(),
#                         RNNEncoderParams(hidden_size=[50, 100])]
