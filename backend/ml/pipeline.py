import logging

import torch.nn.functional as F


logger = logging.getLogger(__name__)


class Pipeline:
    """
    Pipeline combines a trained retrieval model and stance detector.
    """
    def __init__(self, retriever, detector) -> None:
        self._retriever = retriever
        self._detector = detector

    def __call__(self, sentences, misconceptions):
        # Step 1. Retrieve most relevant misconception.
        predictions = self._retriever.predict(sentences, misconceptions)
        logger.debug('Retrieval output: %s', predictions)

        # Step 2. Perform stance detection.
        # TODO: Support multiple relevant misconceptions.
        top_prediction = predictions['predictions'][0]

        output_dict = {
            'input': sentences,
            'predictions': [],
        }
        if top_prediction['misinformation_score'] < 0.4:
            output_dict['relevant'] = False
        else:
            output_dict['relevant'] = True
            # Compute forced label
            misconception = top_prediction['misinformation']
            label_scores = self._detector(
                [sentences],
                [misconception.pos_variations[0]],  # TODO: Use the correct sentence.
            )
            label_id = label_scores[:,:-1].argmax().item()
            label = 'agrees' if label_id == 0 else 'disagrees'
            output_dict['predictions'].append({
                'label': label,
                'misinformation': misconception,
            })
        return output_dict

