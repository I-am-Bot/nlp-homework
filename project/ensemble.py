from scipy import stats
import numpy as np
from sklearn_crfsuite import metrics


class Ensemble:

    def __init__(self, predictions, tag_to_index={'PADword': 0, 'O': 1, 'B-ORG': 2, 'B-MISC': 3, 'B-PER': 4, 'I-PER': 5, 
                                                  'B-LOC': 6,'I-ORG': 7, 'I-MISC': 8, 'I-LOC': 9}, 
                 index_to_tag={0: 'PADword', 1: 'O', 2: 'B-ORG', 3: 'B-MISC', 4: 'B-PER', 5: 'I-PER', 
                               6: 'B-LOC', 7: 'I-ORG', 8: 'I-MISC', 9: 'I-LOC'}, 
                 tags_without_o=['B-ORG', 'B-MISC', 'B-PER', 'I-PER', 'B-LOC', 'I-ORG', 'I-MISC', 'I-LOC'], 
                 best_evaluator=False):
        """
        Takes in a list of prediction lists, forward and backward index to tag conversion dictionaries, and
        a list of labels. It generates and stores an evaluation in 'self.report'
        :param predictions: List of Lists of lists with an inner list format of (word, tag, prediction)
        :param tag_to_index: Dictionary where the key:value pairs are tag:index
        :param index_to_tag: Dictionary where the key: value pairs are index:tag
        :param tages_without_o: List of labels to use when scoring.
        :param best_evaluator: Boolean value for whether or not the first list of predictions
        in the prediction list should be used when all the predictions disagree.
        """
        # create array for numeric values of votes
        ensemble_array = np.zeros((len(y_test), len(predictions)))
        for row in range(ensemble_array.shape[0]):
            for col in range(ensemble_array.shape[1]):
                ensemble_array[row][col] = tag_to_index[predictions[col][row][2]]

        prediction_votes = stats.mode(ensemble_array, axis=1)

        ensemble_pred_y = []
        ensemble_true_y = []

        for idx, vote in enumerate(prediction_votes.mode):
            if (best_evaluator & prediction_votes.count[idx] == 1):
                ensemble_pred_y.append(index_to_tag[int(ensemble_array[idx][0])])
            else:
                ensemble_pred_y.append(index_to_tag[int(vote)])
            ensemble_true_y.append(predictions[0][idx][1])

        self.report = classification_report(ensemble_pred_y, ensemble_true_y, labels=tags_without_o)
