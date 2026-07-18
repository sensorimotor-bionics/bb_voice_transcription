import numpy as np
from nemo.collections.asr.models import SortformerEncLabelModel, EncDecSpeakerLabelModel

enc_model = EncDecSpeakerLabelModel.from_pretrained(model_name="titanet_small").eval()