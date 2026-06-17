from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit

def build_pipeline(n_features_out: int = 12) -> Pipeline:
    """
    Constructs the standard Scikit-learn Pipeline incorporating imputation, Z-scaling, 
    ANOVA pre-filtering, and Sequential Feature Selection.
    
    Args:
        n_features_out (int): Number of features for the final model to select.
        
    Returns:
        Pipeline: The un-fitted Scikit-learn Pipeline.
    """
    # 1. Imputer: Fills remaining NaNs (e.g., very start of the dataset) with median.
    imputer = SimpleImputer(strategy='median')
    
    # 2. Z-Scaler: Normalizes variables so coefficients are accurately scaled and comparable.
    scaler = StandardScaler()
    
    # 3. ANOVA pre-filter: Selects top 40 features to prevent SFS from taking hours on massive global macro universe.
    anova = SelectKBest(score_func=f_classif, k=40)
    
    # 4. Logistic Regression instance for SFS and the final model
    logreg = LogisticRegression(class_weight='balanced', solver='liblinear', random_state=42)
    
    # 5. Sequential Feature Selector
    cv = TimeSeriesSplit(n_splits=3)
    sfs = SequentialFeatureSelector(logreg, n_features_to_select=12, cv=cv, n_jobs=-1)
    
    # Build the strict pipeline
    pipeline = Pipeline([
        ('imputer', imputer),
        ('scaler', scaler),
        ('anova', anova),
        ('sfs', sfs),
        ('clf', logreg)
    ])
    
    return pipeline
