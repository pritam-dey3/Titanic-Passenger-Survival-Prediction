from sklearn.metrics._scorer import _BaseScorer

class WeightedScorer(_BaseScorer):
    def __init__(self, *args, **kwargs):
        self.get_weights = kwargs.pop('rule')
        super().__init__(*args, **kwargs)
    def _score(self, method_caller, estimator, X, y_true, sample_weight=None):
        y_pred = method_caller(estimator, "predict", X)
        sample_weight = self.get_weights(X, y_true)
        return self._sign * self._score_func(y_true, y_pred,
                                                 sample_weight=sample_weight,
                                                 **self._kwargs)