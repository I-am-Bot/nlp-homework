from scipy import stats
import numpy as np
from sklearn_crfsuite import metrics


class Ensemble:

    def __init__(predictions, tag_to_index, index_to_tag, test_y, tags_without_o, best_evaluator=False,
                 flat=False):
        """
        Takes in a list of prediction lists, forward and backward index to tag conversion dictionaries, and
        a list of labels. It generates and stores an evaluation in 'self.report'
        :param tag_to_index: dictionary where the key:value pairs are tag:index
        :param index_to_tag: dictionary where the key: value pairs are index:tag
        :param test_y: The true labels for checking the quality of the predictions
        :param labels: List of labels to use when scoring.
        :param best_evaluator: Boolean value for whether or not the first list of predictions
        in the prediction list should be used when all the predictions disagree.
        :param flat: Boolean value for whether or not the prediction lists are all one dimensions or nested
        lists.
        """

        # create array for numeric values of votes
        ensemble_array = np.zeros((len(y_test), len(predictions)))
        for row in range(ensemble_array.shape[0]):
            for col in range(ensemble_array.shape[1]):
                ensemble_array[row][col] = tag_to_index(predictions[col][row])

        prediction_votes = stats.mode(ensemble_array, axis=1)

        ensemble_pred_y = []

        for idx, vote in enumerate(prediction_votes.mode):
            if (best_evaluator & prediction_votes.count[idx] == 1):
                ensemble_pred_y.append(index_to_tag[int(ensemble_array[idx][0])])
            else:
                ensemble_pred_y.append(index_to_tag[int(vote)])

        if flat:
            self.report = metrics.flat_classification_report(ensemble_pred_y, test_y, labels=tags_without_o)
        else:
            self.report = classification_report(ensemble_pred_y, ensemble_true_y)
