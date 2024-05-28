from models.multimodal import *
from models.text import *

SUPPORTED_MODELS = {
    'BERTMLP': BERTMLP,
    'EfficientPunct': EfficientPunct,
    'EfficientPunctBERT': EfficientPunctBERT,
    'EfficientPunctTDNN': EfficientPunctTDNN,
    'LengthConditional': LengthConditional,
    'UniPunc': UniPunc
}