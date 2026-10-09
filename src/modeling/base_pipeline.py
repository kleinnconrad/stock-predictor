from typing import Tuple

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, SequentialFeatureSelector, VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit


class InsufficientHistoryError(ValueError):
    """Raised when a purged time-series split cannot leave enough training rows."""


def purged_time_series_cv(n_samples: int, n_splits: int, gap: int, min_train_rows: int,
                          allow_unpurged: bool = False) -> Tuple[TimeSeriesSplit, bool]:
    """
    Builds an expanding-window TimeSeriesSplit with a gap between train and test rows.

    The target is a forward return over `gap` trading days, so the last `gap` training
    labels overlap with prices inside the following test window. Dropping those rows
    (the gap) prevents that leakage. The number of splits is reduced until the first
    training fold keeps at least `min_train_rows` rows.

    Args:
        n_samples (int): Number of rows to split.
        n_splits (int): Preferred number of splits (reduced down to 2 if needed).
        gap (int): Rows dropped between each training fold and its test fold.
        min_train_rows (int): Minimum size of the first training fold.
        allow_unpurged (bool): If no purged split is feasible, return an unpurged
            split instead of raising.

    Returns:
        Tuple[TimeSeriesSplit, bool]: The splitter and whether it is purged.

    Raises:
        InsufficientHistoryError: If no purged split is feasible and `allow_unpurged` is False.
    """
    for k in range(n_splits, 1, -1):
        test_size = n_samples // (k + 1)
        first_train_rows = n_samples - k * test_size - gap
        if test_size > 0 and first_train_rows >= min_train_rows:
            return TimeSeriesSplit(n_splits=k, gap=gap), True
    if allow_unpurged:
        return TimeSeriesSplit(n_splits=n_splits), False
    raise InsufficientHistoryError(
        f"{n_samples} rows cannot form a purged split (gap={gap}, min_train_rows={min_train_rows})."
    )


def build_pipeline(n_features_out: int = 12, anova_k: int = 40, sfs_cv=None) -> Pipeline:
    """
    Constructs the standard Scikit-learn Pipeline incorporating imputation, Z-scaling,
    ANOVA pre-filtering, and Sequential Feature Selection.

    Args:
        n_features_out (int): Number of features for the final model to select.
        anova_k (int): Number of features kept by the ANOVA pre-filter.
        sfs_cv: Cross-validation splitter used by the Sequential Feature Selector.
            Defaults to an unpurged 5-fold TimeSeriesSplit.

    Returns:
        Pipeline: The un-fitted Scikit-learn Pipeline.
    """
    # 1. Imputer: Fills remaining NaNs (e.g., very start of the dataset) with median.
    imputer = SimpleImputer(strategy='median')

    # 2. Variance Threshold: Drops constant features (0 variance) to prevent divide by zero errors in ANOVA.
    var_thresh = VarianceThreshold()

    # 3. Z-Scaler: Normalizes variables so coefficients are accurately scaled and comparable.
    scaler = StandardScaler()

    # 4. ANOVA pre-filter: Keeps the top `anova_k` features so SFS stays tractable on the large macro universe.
    anova = SelectKBest(score_func=f_classif, k=anova_k)

    # 5. Logistic Regression instance for SFS and the final model
    logreg = LogisticRegression(class_weight='balanced', solver='liblinear', random_state=42)

    # 6. Sequential Feature Selector
    cv = sfs_cv if sfs_cv is not None else TimeSeriesSplit(n_splits=5)
    sfs = SequentialFeatureSelector(logreg, n_features_to_select=n_features_out, cv=cv, n_jobs=-1)

    # Build the strict pipeline
    pipeline = Pipeline([
        ('imputer', imputer),
        ('var_thresh', var_thresh),
        ('scaler', scaler),
        ('anova', anova),
        ('sfs', sfs),
        ('clf', logreg)
    ])

    return pipeline
