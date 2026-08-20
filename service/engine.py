import math
import numpy as np
import torch


class TurnTakingEngine:
    def __init__(
        self,
        model,
    ):
        """
        model: TurnModel (model.py)
        """
        self.model = model
        self.device = model.device
        self.contexts = None

    def process(self, audio: np.ndarray, asr_text=None, asr_final=False):
        """
        audio: np.float32, shape [N]
        return:
            None | state_token(str)
        """
        if self.contexts is None:
            self.model.reset()
        else:
            self.model.restore_runtime(self.contexts)

        result = self.model.process(audio, asr_text=asr_text, asr_final=asr_final)
        self.contexts = self.model.snapshot_runtime()

        return result
