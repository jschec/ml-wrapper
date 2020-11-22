from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFECV, SelectFromModel, VarianceThreshold, SelectKBest, StratifiedKFold, chi2
from sklearn.svm import SVC, LinearSVC

encoding_strats = {
    "onehot", OneHotEncoder(handle_unknown='ignore')
}

scaling_strats = {
    'standard_scaler': StandardScaler(),
    'min_max_scaler': MinMaxScaler()
}

imputing_strats = {
    'mean': SimpleImputer(strategy='mean'), 
    'median': SimpleImputer(strategy='median'), 
    'most_frequent': SimpleImputer(strategy='most_frequent'), 
    'constant_number_zero': SimpleImputer(strategy='constant', fill_value=0),
    'constant_cat_missing': SimpleImputer(strategy='constant', fill_value="missing")
}

feature_selection_strats = {
    'rfecv': RFECV(estimator=SVC(kernel="linear"), step=1, cv=StratifiedKFold(2), scoring='accuracy'),
    'SelectKBest': SelectKBest(score_func=chi2, k=30),
    'VarianceThreshold': VarianceThreshold(threshold=(.8 * (1 - .8))),
    'SelectFromModel': SelectFromModel(LinearSVC(C=0.01, penalty='l1', dual=False, max_iter=2000))
}

decomposition_strats = {
    'KernelPCA': KernelPCA(n_components=7, kernel='linear'),
    'PCA': PCA(n_components=3),
    'TruncatedSVD', TruncatedSVD(n_components=2),
    'SparsePCA', SparsePCA(n_components=2, normalize_components=True)
}

#other_steps = [
        #('decomposition', KernelPCA(n_components=7, kernel='linear')),
        #('decomposition', PCA(n_components=3)),
        #('decomposition', TruncatedSVD(n_components=2))
        #('decomposition', SparsePCA(n_components=2, normalize_components=True)),
    #] 