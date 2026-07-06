import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from pyod.models.hdbscan import HDBSCAN
from pyod.models.knn import KNN
from pyod.models.pca import PCA
from pyod.models.iforest import IForest

import numpy as np

rng = np.random.default_rng(42)
X_train = rng.normal(size=(500, 8))
X_test = rng.normal(size=(100, 8))

for Detector in [PCA, KNN, IForest, HDBSCAN]:
    scores = Detector().fit(X_train).decision_function(X_test)
    print(f"{Detector.__name__:8s}  max-Score = {scores.max():.2f}")


