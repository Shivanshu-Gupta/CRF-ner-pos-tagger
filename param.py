import attr
import torch
from collections.abc import Mapping
from itertools import product
from copy import deepcopy

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

class Settings(list):
    def __init__(self, *args, **kwargs):
        super(Settings, self).__init__(args[0])

class Parameters(DictDataClass):
    def get_settings(self):
        keys = []
        value_lists = []
        for k, v in self.items():
            if isinstance(v, Parameters):
                value_lists.append(v.get_settings())
                keys.append(k)
            # elif isinstance(v, Settings):
            elif isinstance(v, list):
                assert(len(v) > 0)
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
    embedding_dim: int = 50
    # embedding_path: str = "embeddings/glove.twitter.27B/glove.twitter.27B.50d.bin"

@attr.s(auto_attribs=True)
class EncoderParams(Parameters):
    type: str = None

@attr.s(auto_attribs=True)
class RNNEncoderParams(EncoderParams):
    type: str = 'torch.nn.LSTM'
    hidden_size: int = 100
    num_layers: int = 1
    bias: bool = True
    dropout: float = 0
    bidirectional: bool = True

@attr.s(auto_attribs=True)
class ProjectionParams(Parameters):
    type: str = 'torch.nn.Linear'
    in_features: int = 50
    out_features: int = 13

@attr.s(auto_attribs=True)
class ModelParams(Parameters):
    type: str = 'simple_tagger.SimpleTagger'
    embeddings: EmbeddingParams = EmbeddingParams()
    encoder: EncoderParams = EncoderParams()
    # encoder: EncoderParams = RNNEncoderParams()
    tag_projection: ProjectionParams = ProjectionParams()

@attr.s(auto_attribs=True)
class OptimizerParams(Parameters):
    type: str = 'torch.optim.Adam'
    lr: float = 0.001

@attr.s(auto_attribs=True)
class TrainingParams(Parameters):
    batch_size: int = 32
    num_epochs: int = 40
    optimizer: OptimizerParams = OptimizerParams()

@attr.s(auto_attribs=True)
class TaggingParams(Parameters):
    random_seed: int = 42
    dataset: str = 'ner'
    train_dataset: DataParams = DataParams(data_path="data/twitter_train.ner")
    validation_dataset: DataParams = DataParams(data_path="data/twitter_dev.ner")
    model: ModelParams = ModelParams()
    training: TrainingParams = TrainingParams()

    def __attrs_post_init__(self):
        if self.dataset == 'ner':
            self.train_dataset.data_path = "data/twitter_train.ner"
            self.validation_dataset.data_path = "data/twitter_dev.ner"
            self.model.tag_projection.out_features = 22
        else:
            self.train_dataset.data_path = "data/twitter_train.pos"
            self.validation_dataset.data_path = "data/twitter_dev.pos"
            self.model.tag_projection.out_features = 13

params = TaggingParams()
# params.model.encoder = [EncoderParams(),
#                         RNNEncoderParams(hidden_size=[50, 100])]

# import pandas as pd
# pd.json_normalize(attr.asdict(params), sep='.')
