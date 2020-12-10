from logging import Logger

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MultiLabelBinarizer


def log_performance(
    X_test: np.ndarray,
    y_test_binarized: np.ndarray,
    model: BaseEstimator,
    binarizer: MultiLabelBinarizer,
    logger: Logger,
) -> None:
    """Logs performance of the model to the log file"""
    y_test_pred_binarized = model.predict(X_test)
    logger.info("-" * 80)
    logger.info("**EVALUATION\nClassification Report \n**")
    logger.info(
        classification_report(
            y_test_binarized, y_test_pred_binarized, target_names=binarizer.classes_, zero_division=1
        )
    )
    logger.info("\nAccuracy Score: {}".format(accuracy_score(y_test_binarized, y_test_pred_binarized)))
