from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomTreesEmbedding, RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
#from sklearn.linear_model import LinearRegression

classification_models = {
    'logisticRegression': ('classifier', LogisticRegression(solver='lbfgs', max_iter=1000)),
    'linearSVC': ('classifier', LinearSVC(tol=1e-5)),
    'linearSVC2': ('classifier', LinearSVC(C=0.01, penalty='l1', dual=False, max_iter=2000)),
    'SVC': ('classifier', SVC(gamma='auto')),
    'SVC2': ('classifier', SVC(gamma=2, C=1)),
    'naiveBayes': ('classifier', GaussianNB()),
    'Kneighbors':  ('classifier', KNeighborsClassifier(n_jobs=-1)),
    'extraTreesClassifier':  ('classifier', ExtraTreesClassifier()),
    'randomForestClassifier': ('classifier', RandomForestClassifier(max_depth=5, max_features=1, n_estimators=100)),
    'tree': ('classifier', DecisionTreeClassifier(max_depth=5)),
    'LDA': ('classifier', LinearDiscriminantAnalysis()),
    'AdaBoost': ('classifier', AdaBoostClassifier()),
    'QDA': ('classifier', QuadraticDiscriminantAnalysis()),
    'NN': ('classifier', MLPClassifier(alpha=1, max_iter=1000)),
    'Gaussian': ('classifier', GaussianProcessClassifier(1.0 * RBF(1.0)))
}

clustering_models = {
    'kmeans': ('clustering', KMeans(n_jobs=-1))
}