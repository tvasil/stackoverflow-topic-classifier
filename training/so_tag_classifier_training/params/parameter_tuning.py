import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.multioutput import ClassifierChain

search_space = [
    {
        "tfidf__min_df": np.arange(5, 100),
        "tfidf__max_df": np.arange(0.01, 0.98, step=0.01),
        "clf": [
            ClassifierChain(
                LogisticRegression(random_state=42, dual=False, solver="liblinear", max_iter=1000), cv=3
            )
        ],
        "clf__estimator__C": np.arange(0.000001, 1000, step=0.01),
        "clf__estimator__penalty": ["l1", "l2"],
    }
]

scoring = {"f1": make_scorer(f1_score, average="weighted"), "average_precision": "average_precision"}
