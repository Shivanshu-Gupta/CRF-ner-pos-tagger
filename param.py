import attr
import cattr
import pandas as pd
from collections.abc import Mapping
from typing import Union
from itertools import product
from copy import deepcopy

from util import nest_dict

datasets = ['ner', 'pos']
models = ['simple', 'crf']


@attr.s(auto_attribs=True)
class DictDataClass(Mapping):
    """Allow dict-like access to attributes using ``[]`` operator in addition to dot-access."""

    def __iter__(self):
        return iter(vars(self))

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __len__(self):
        return len(vars(self))

    def to_flattened_dict(self, sep='.'):
        _d = pd.json_normalize(attr.asdict(self), sep=sep).iloc[0].to_dict()
        # hack to fix tune.grid_search() values as they are themselves dicts
        d = {}
        for k, v in _d.items():
            if k.endswith('grid_search'):
                parts = k.split(sep)
                d[sep.join(parts[:-1])] = {'grid_search': v}
            else:
                d[k] = v
        return d

    @classmethod
    def from_flattened_dict(cls, d: dict):
        return cls.from_dict(nest_dict(d))

    @classmethod
    def from_dict(cls, d: dict):
        converter = cattr.Converter()
        return converter.structure(d, cls)


class Parameters(DictDataClass):
    def get_settings(self):
        keys = []
        value_lists = []
        for k, v in self.items():
            if isinstance(v, Parameters):
                value_lists.append(v.get_settings())
                keys.append(k)
            elif isinstance(v, list):
                assert (len(v) > 0)
                if isinstance(v[0], Parameters):
                    value_lists.append([s for _v in v for s in _v.get_settings()])
                else:
                    value_lists.append(v)
                keys.append(k)
        settings = []
        for values in product(*value_lists):
            for k, v in zip(keys, values):
                setattr(self, k, v)
            settings.append(deepcopy(self))
        return settings


@attr.s(auto_attribs=True)
class DataParams(Parameters):
    data_path: str = None
    max_length = 30


@attr.s(auto_attribs=True)
class EmbeddingParams(Parameters):
    embedding: Union[int, str] = 50


# @attr.s(auto_attribs=True)
# class EncoderParams(Parameters):
#     type: str = ''

@attr.s(auto_attribs=True)
class EncoderParams(Parameters):
    type: str = ''
    hidden_size: int = 100
    num_layers: int = 1
    bias: bool = True
    dropout: float = 0
    bidirectional: bool = True


@attr.s(auto_attribs=True)
class ProjectionParams(Parameters):
    type: str = 'torch.nn.Linear'
    # in_features: int = 50
    out_features: int = 13


@attr.s(auto_attribs=True)
class ModelParams(Parameters):
    type: str = 'simple_tagger.SimpleTagger'
    embedding_param: str = '50'
    # embeddings: EmbeddingParams = attr.ib(default=attr.Factory(lambda: EmbeddingParams()))
    encoder: EncoderParams = attr.ib(default=attr.Factory(lambda: EncoderParams()))
    tag_projection: ProjectionParams = attr.ib(default=attr.Factory(lambda: ProjectionParams()))


@attr.s(auto_attribs=True)
class OptimizerParams(Parameters):
    type: str = 'torch.optim.Adam'
    lr: float = 0.001


@attr.s(auto_attribs=True)
class SGDOptimizerParams(OptimizerParams):
    type: str = 'torch.optim.SGD'
    lr: float = 0.001
    momentum: float = 0.1


@attr.s(auto_attribs=True)
class TrainingParams(Parameters):
    num_epochs: int = 20
    optimizer: OptimizerParams = attr.ib(default=attr.Factory(lambda: OptimizerParams()))


@attr.s(auto_attribs=True)
class TaggingParams(Parameters):
    random_seed: int = 42
    gpu_idx: int = -1
    dataset: str = 'ner'
    train_dataset: DataParams = attr.ib(default=attr.Factory(lambda: DataParams(data_path="data/twitter_train.ner")))
    validation_dataset: DataParams = attr.ib(default=attr.Factory(lambda: DataParams(data_path="data/twitter_dev.ner")))
    train_batch_size: int = 32
    val_batch_size: int = 64
    model: ModelParams = attr.ib(default=attr.Factory(lambda: ModelParams()))
    training: TrainingParams = attr.ib(default=attr.Factory(lambda: TrainingParams()))
    search_name: str = ''
    is_grid: bool = True

    def __attrs_post_init__(self):
        if self.dataset == 'ner':
            self.train_dataset.data_path = "data/twitter_train.ner"
            self.validation_dataset.data_path = "data/twitter_dev.ner"
            self.model.tag_projection.out_features = 22
        else:
            self.train_dataset.data_path = "data/twitter_train.pos"
            self.validation_dataset.data_path = "data/twitter_dev.pos"
            self.model.tag_projection.out_features = 13

# params = TaggingParams()
# params.model.encoder = [EncoderParams(),
#                         RNNEncoderParams(hidden_size=[50, 100])]
