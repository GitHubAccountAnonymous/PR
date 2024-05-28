from feats.multimodal import MultimodalFeaturizerKaldi
from feats.text import BERTTokenizerFeaturizer

SUPPORTED_FEATURIZERS = {
    'BERTMLP': BERTTokenizerFeaturizer,
    ("Kaldi", "BERTMLP"): MultimodalFeaturizerKaldi
}