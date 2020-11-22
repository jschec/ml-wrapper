import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, consensus_score, roc_curve, confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sql_client import *
import numpy as np
from sklearn.ensemble import RandomTreesEmbedding, RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV, SelectFromModel
from sklearn.feature_selection import VarianceThreshold
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster.bicluster import SpectralBiclustering
from sklearn.datasets import make_biclusters
from sklearn.datasets import samples_generator as sg
import matplotlib as mpl
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import model_selection
from sklearn.neural_network import MLPClassifier
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS, cluster_optics_dbscan
import umap

#SGDClassifier
#kernalApproximation

#clustering models
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.cluster import MeanShift, estimate_bandwidth

from sklearn.svm import SVC
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit


from sklearn.decomposition import PCA, TruncatedSVD, SparsePCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn import cross_validation
from sklearn import tree
from sklearn.datasets import load_digits
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

 
#OLD... need to update
	
def gene_level_pipeline():
    """
	db = SQLiteDB('./gene_master.db')
	result = db.execute_select_stmt(
		'SELECT table_a.gene_name, * \
        FROM (SELECT gene_name FROM c2_pathway_pca_t UNION SELECT gene_short_name AS gene_name FROM master_allen_data_t UNION SELECT distinct_gene_name AS gene_name FROM master_insilico_t) AS table_a\
        LEFT JOIN c2_pathway_pca_t ON c2_pathway_pca_t.gene_name = table_a.gene_name \
        LEFT JOIN master_allen_data_t ON master_allen_data_t.gene_short_name = table_a.gene_name \
        LEFT JOIN master_insilico_t ON master_insilico_t.distinct_gene_name = table_a.gene_name'
	)
    """
    cluster_number = 3
   # db = SQLiteDB('../datasets/sqlite_dbs/gene_in_silico.db')
    df1 = pd.read_csv('../datasets/pathway/c2_pathway_pca_results.csv')
    df1 = df1[df1.columns[1:]]
    df2 = pd.read_csv('../datasets/in-silico/master_in_silico.csv')
    df2 = df2[df2.columns[1:]]
    #df2 = db.execute_select_stmt('SELECT "#HGNC ID" AS distinct_gene_name, "Score", "STRING-combined score", "ExAC-pRec", "STRING-experimental score", "ExAC-missense z-score",  "PhyloP at 5\'-UTR", "STRING-textmining score", "Number donor/number synonymous", "mRNA half-life->10h", "lda score" FROM domino_annotations_t')
    df3 = pd.read_csv('../datasets/single_cell/allen_single_cell_pca.csv')
    df3 = df3[df3.columns[1:]]
    df4 = pd.read_csv('../datasets/dbdb_pathways/dbdb_gene_present.csv')
    df4 = df4[df4.columns[1:]]

    #print(df3)
    joined_df = df1.set_index('distinct_gene_name').join(df2.set_index('distinct_gene_name'))
    joined_df = joined_df.join(df3.set_index('distinct_gene_name'), on='distinct_gene_name')
    joined_df = joined_df.join(df4.set_index('distinct_gene_name'), on='distinct_gene_name')
    #fig, ax = plt.subplots()

    #db = SQLiteDB('../datasets/sqlite_dbs/gene_in_silico.db')
    #db = SQLiteDB('../datasets/sqlite_dbs/gene_master.db')
    #result = db.execute_select_stmt('SELECT * FROM master_allen_data_t')
    #result = pd.read_csv('../datasets/pathway/processed_c2_pathway_sparse_matrix.csv')
    joined_df.apply(pd.to_numeric, errors='coerce')
    joined_df = joined_df.fillna(0)
    print('transforming...')
    dbdb_gene_names_list = joined_df['dbdb_presence']
    joined_df = joined_df[joined_df.columns[0:-1]]
    scaled_result = MinMaxScaler().fit_transform(joined_df)
    
    print(dbdb_gene_names_list)
    print(scaled_result)
    
    #X_embedded = umap.UMAP(random_state=42).fit_transform(scaled_result)
    print('creating first round of clusters...')
    kmeans = KMeans(n_clusters=cluster_number).fit(scaled_result)
    labels = kmeans.labels_
    print('creating second round of clusters...')
    optics = OPTICS(min_samples=50, xi=.05, min_cluster_size=.05).fit(scaled_result)
    optics_labels = optics.labels_
    #optics_labels = labels
    print('creating manifold...')
    X_embedded = TSNE(n_components=2).fit_transform(scaled_result)
    #dbscan = DBSCAN(eps=0.3, min_samples=10).fit(scaled_result)
    #dbscan_labels = dbscan.labels_
    #final_df = pd.DataFrame({'tsne1': X_embedded[:, 0], 'tsne2': X_embedded[:, 1], 'tsne3': X_embedded[:, 2], 'labels': labels})
    final_df = pd.DataFrame({'tsne1': X_embedded[:, 0], 'tsne2': X_embedded[:, 1], 'labels': labels, 'labels2': optics_labels, 'dbdb_present': dbdb_gene_names_list})
    #final_df.to_csv('./clustering_data.csv')
    #final_df2 = pd.DataFrame({'tsne1': X_embedded[:, 0], 'tsne2': X_embedded[:, 1], 'labels': optics_labels})
    #final_df = pd.DataFrame({'tsne1': joined_df['pca1'], 'tsne2': joined_df['pca2'], 'labels': labels, 'labels2': labels})
    colors = ['#4EACC5', '#FF9C34', '#4E9A06']
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    fig, axs = plt.subplots(1, 3, figsize=(16, 6))
    #axs[0].scatter(final_df['tsne1'], final_df['tsne2'], c='black', s=4)
    first_scatter_data = final_df[final_df['dbdb_present']==0]
    axs[0].scatter(first_scatter_data['tsne1'], first_scatter_data['tsne2'], c='black', s=1)
    first_scatter_data = final_df[final_df['dbdb_present']==1]
    axs[0].scatter(first_scatter_data['tsne1'], first_scatter_data['tsne2'], c='blue', s=1)

    for label_num in range(cluster_number):
        print(label_num)
        scatter_data = final_df[final_df['labels']==label_num]
        axs[1].scatter(scatter_data['tsne1'], scatter_data['tsne2'], c=colors[label_num], s=1)
    
    axs[2].scatter(final_df['tsne1'], final_df['tsne2'], c=final_df['labels2'], s=1)
        #ax.scatter(scatter_data['tsne1'], scatter_data['tsne2'], scatter_data['tsne3'], c=colors[label_num], s=7)

    axs[0].set_title('No Clustering')
    axs[1].set_title('Kmeans')
    axs[2].set_title('OPTICS')
    plt.tight_layout()
    #cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)
    #ax = sns.scatterplot(x='tsne1', y='tsne2', hue="labels", palette=cmap, data=final_df)
    #plt.style.use('ggplot')
    #plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c='black', s=7)
    plt.savefig('./try2.png')


    """
    result = pd.read_csv('../datasets/pathway/c2_pathway_sparse_matrix.csv')
    print('result')
    print(result)
    print(result.shape)
    #distinct_gene_name
    #result = result[result.columns]
    distinct_gene_name = result[result.columns[-1]]
    print('columns', result.columns)
    result = result[result.columns[1:-1]]
    print()
    print('new result')
    print(result)
    print(result.shape)
    result = result.apply(pd.to_numeric, errors='coerce')
    result = result.fillna(0)
    print('new result1')
    print(result)
    print(result.shape)
    
    #X_embedded = TSNE(n_components=2).fit_transform(result)
    pca1 = PCA(n_components=3)
    #scaled_result = StandardScaler().fit_transform(result)
    #feature_scaled_pca = pca1.fit_transform(scaled_result)
    feature_scaled_pca = pca1.fit_transform(result)
    

    new_df = pd.DataFrame({'distinct_gene_name': distinct_gene_name, 'pca1': feature_scaled_pca[:, 0], 'pca2': feature_scaled_pca[:, 1], 'pca3': feature_scaled_pca[:, 2]})
    new_df.to_csv('c2_pathway_pca_results.csv')
    """
    
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(feature_scaled_pca[:, 0], feature_scaled_pca[:, 1], feature_scaled_pca[:, 2])


    #plt.style.use('ggplot')
    #plt.scatter(feature_scaled_pca[:, 0], feature_scaled_pca[:, 1], c='black')
    #plt.savefig('./pca_c2_pathway3d.png')
    #plt.scatter(feature_scaled_pca[:, 2], feature_scaled_pca[:, 3], c='black', s=7)
    #plt.savefig('./pca_plot_new2.png')


    #gene_level_ml_pipeline = ml_executor(data_cols, gene_cat_attributes, table_data=result)
    #gene_level_ml_pipeline.configure_transformation_strategy(num_imputer_strategy='constant_number_zero', num_scaler_stategy='min_max_scaler', cat_imputer_strategy='constant_cat_missing', cat_encoder_strategy='onehot')
    #gene_level_ml_pipeline.implement_ml_model(model_type='clustering', model_name='kmeans', other_steps=other_steps)


def violin_plot():
    variant_num_attributes = [
		#'rmsk2', 'genomicSuperDups2', 'cosmic88_coding2', #coded columns
		#'Start', 'End', 
		'SIFT_score', #'SIFT_converted_rankscore', 
        'Polyphen2_HDIV_score', #'Polyphen2_HDIV_rankscore',
        'Polyphen2_HVAR_score', #'Polyphen2_HVAR_rankscore', 
		'LRT_score', #'LRT_converted_rankscore', 
		'MutationTaster_score', #'MutationTaster_converted_rankscore', 
        'MutationAssessor_score', #'MutationAssessor_score_rankscore',
        'FATHMM_score', #'FATHMM_converted_rankscore', 
        'PROVEAN_score', #'PROVEAN_converted_rankscore', 
        'VEST3_score', #'VEST3_rankscore', 
        'MetaSVM_score', #'MetaSVM_rankscore', 
		'MetaLR_score', #'MetaLR_rankscore', 
        #'M-CAP_score', 'M-CAP_rankscore', 
        'REVEL_score', #'REVEL_rankscore', 
        'MutPred_score', #'MutPred_rankscore', 
        'CADD_phred', #'CADD_raw', 'CADD_raw_rankscore',  
        'DANN_score', #'DANN_rankscore', 
		'fathmm-MKL_coding_score', #'fathmm-MKL_coding_rankscore',
        'Eigen-raw', 'Eigen-PC-raw', 
		'GenoCanyon_score', #'GenoCanyon_score_rankscore', 
		'integrated_fitCons_score', #'integrated_fitCons_score_rankscore', 'integrated_confidence_value', 
        'GERP++_RS', #'GERP++_RS_rankscore', 
        'phyloP100way_vertebrate', #'phyloP100way_vertebrate_rankscore', 
        'phyloP20way_mammalian', #'phyloP20way_mammalian_rankscore', 
        'phastCons100way_vertebrate', #'phastCons100way_vertebrate_rankscore', 
        'phastCons20way_mammalian', #'phastCons20way_mammalian_rankscore',
        'SiPhy_29way_logOdds', #'SiPhy_29way_logOdds_rankscore', 
        'MCAP13', 
        'REVEL', 
		'AF', 'AF_popmax', 
        #'AF_male', 'AF_female', 'AF_raw', 'AF_afr', 'AF_sas',
        #'AF_amr', 'AF_eas', 'AF_nfe', 'AF_fin', 'AF_asj', 'AF_oth', 'non_topmed_AF_popmax', 'non_neuro_AF_popmax', 'non_cancer_AF_popmax', 'controls_AF_popmax',
        'AF2', 'AF2_popmax', 
        #'AF2_male', 'AF2_female', 'AF2_raw', 'AF2_afr', 'AF2_amr', 'AF2_eas', 'AF2_nfe', 'AF2_fin', 'AF2_asj', 'AF2_oth', 'non_topmed_AF2_popmax',
        #'non_neuro_AF2_popmax', 'controls_AF2_popmax'                
        #'non_cancer_AF2_popmax',
        #'AF2_sas',
         'clinsig_2_bin', 'clinsig_4_bin',
    ]

    #columns = variant_num_attributes
    #mpl.rcParams['figure.figsize'] = (16, 6)
    variant_cat_attributes = []
    where_stms = 'WHERE "ExonicFunc.refGene" LIKE "%nonsynonymous%"'
    db = sqlitedb_connection("../clinvar.db")
    results = db.execute_query(columns_to_use=variant_num_attributes, table_to_query='master_annotated_variants_coded_final', target_feature=0, where_stmt=where_stms)
    results = results.apply(pd.to_numeric, errors='coerce') 
    fig, axes = plt.subplots(7, 5, figsize=(20, 30))
    #figsize=(6, 17)
    #17
    sns.violinplot(x=results["SIFT_score"], ax=axes[0, 0])
    #axes[0, 0].set_title('SIFT_score')
    sns.violinplot(x=results["Polyphen2_HDIV_score"], ax=axes[0, 1])
    #axes[0, 1].set_title('Polyphen2_HDIV_score')
    sns.violinplot(x=results["Polyphen2_HVAR_score"], ax=axes[0, 2])
    #axes[0, 2].set_title('Polyphen2_HVAR_score')
    sns.violinplot(x=results["LRT_score"], ax=axes[0, 3])
    #axes[0, 3].set_title('LRT_score')
    sns.violinplot(x=results["MutationTaster_score"], ax=axes[0, 4])
    #axes[0, 4].set_title('MutationTaster_score')
    sns.violinplot(x=results["MutationAssessor_score"], ax=axes[1, 0])
    #axes[1, 0].set_title('MutationAssessor_score')
    sns.violinplot(x=results["FATHMM_score"], ax=axes[1, 1])
    #axes[1, 1].set_title('FATHMM_score')
    sns.violinplot(x=results["PROVEAN_score"], ax=axes[1, 2])
    #axes[1, 2].set_title('PROVEAN_score')
    sns.violinplot(x=results["VEST3_score"], ax=axes[1, 3])
    #axes[1, 3].set_title('VEST3_score')
    sns.violinplot(x=results["MetaSVM_score"], ax=axes[1, 4])
    #axes[1, 4].set_title('MetaSVM_score')
    sns.violinplot(x=results["MetaLR_score"], ax=axes[2, 0])
    #axes[2, 0].set_title('MetaLR_score')
    sns.violinplot(x=results["REVEL_score"], ax=axes[2, 1])
    #axes[2, 1].set_title('REVEL_score')
    sns.violinplot(x=results["MutPred_score"], ax=axes[2, 2])
    #axes[2, 2].set_title('MutPred_score')
    sns.violinplot(x=results["CADD_phred"], ax=axes[2, 3])
    #axes[2, 3].set_title('CADD_phred')
    sns.violinplot(x=results["DANN_score"], ax=axes[2, 4])
    #axes[2, 4].set_title('DANN_score')
    sns.violinplot(x=results["fathmm-MKL_coding_score"], ax=axes[3, 0])
    #axes[3, 0].set_title('fathmm-MKL_coding_score')
    sns.violinplot(x=results["Eigen-raw"], ax=axes[3, 1])
    #axes[3, 1].set_title('Eigen-raw')
    sns.violinplot(x=results["Eigen-PC-raw"], ax=axes[3, 2])
    #axes[3, 2].set_title('Eigen-PC-raw')
    sns.violinplot(x=results["GenoCanyon_score"], ax=axes[3, 3])
    #axes[3, 3].set_title('GenoCanyon_score')
    sns.violinplot(x=results["integrated_fitCons_score"], ax=axes[3, 4])
    #axes[3, 4].set_title('integrated_fitCons_score')
    sns.violinplot(x=results["GERP++_RS"], ax=axes[4, 0])
    #axes[4, 0].set_title('GERP++_RS')
    sns.violinplot(x=results["phyloP100way_vertebrate"], ax=axes[4, 1])
    #axes[4, 1].set_title('phyloP100way_vertebrate')
    sns.violinplot(x=results["phyloP20way_mammalian"], ax=axes[4, 2])
    #axes[4, 2].set_title('phyloP20way_mammalian')
    sns.violinplot(x=results["phastCons100way_vertebrate"], ax=axes[4, 3])
    #axes[4, 3].set_title('phastCons100way_vertebrate')
    sns.violinplot(x=results["phastCons20way_mammalian"], ax=axes[4, 4])
    #axes[4, 4].set_title('phastCons20way_mammalian')
    sns.violinplot(x=results["SiPhy_29way_logOdds"], ax=axes[5, 0])
    #axes[5, 0].set_title('SiPhy_29way_logOdds')
    sns.violinplot(x=results["MCAP13"], ax=axes[5, 1])
    #axes[5, 1].set_title('MCAP13')
    sns.violinplot(x=results["REVEL"], ax=axes[5, 2])
    #axes[5, 2].set_title('REVEL')
    sns.violinplot(x=results["AF"], ax=axes[5, 3])
    #axes[5, 3].set_title('AF')
    sns.violinplot(x=results["AF_popmax"], ax=axes[5, 4])
    #axes[5, 4].set_title('AF_popmax')
    sns.violinplot(x=results["AF2"], ax=axes[6, 0])
    #axes[6, 0].set_title('AF2')
    sns.violinplot(x=results["AF2_popmax"], ax=axes[6, 1])
    #axes[6, 1].set_title('AF2_popmax')
    sns.violinplot(x=results["clinsig_2_bin"], ax=axes[6, 2])
    #axes[6, 2].set_title('clinsig_2_bin')
    sns.violinplot(x=results["clinsig_4_bin"], ax=axes[6, 3])
    #axes[6, 3].set_title('clinsig_4_bin')
    plt.tight_layout()
    plt.savefig('./feature_variation.png')


	
def variant_level_pipeline():
    variant_num_attributes = [
		#'SIFT_score',
        #'Polyphen2_HDIV_score',
        #'Polyphen2_HVAR_score',
		#'LRT_score',  
		#'MutationTaster_score', 
        'MutationAssessor_score', 
        'FATHMM_score', 
        'PROVEAN_score',  
        'VEST3_score', 
        #'MetaSVM_score', 
		'MetaLR_score', 
        #'REVEL_score', 
        'MutPred_score',
        'CADD_phred',  
        #'DANN_score', 
		#'fathmm-MKL_coding_score', 
        #'Eigen-raw', 'Eigen-PC-raw', 
		#'GenoCanyon_score', 
		#'integrated_fitCons_score',
        #'GERP++_RS',
        #'phyloP100way_vertebrate',
        #'phyloP20way_mammalian',
        #'phastCons100way_vertebrate',
        #'phastCons20way_mammalian',
        #'SiPhy_29way_logOdds',
        'MCAP13', 
        'REVEL', 
		#'AF', 
        'AF_popmax', 
        #'AF2', 'AF2_popmax', 


         #'clinsig_2_bin', 'clinsig_4_bin',
    ]

    columns = variant_num_attributes
    #mpl.rcParams['figure.figsize'] = (16, 6)
    variant_cat_attributes = []
    where_stms = 'WHERE "ExonicFunc.refGene" LIKE "%nonsynonymous%"'
    #db = sqlitedb_connection("./clinvar.db")
    #results = db.execute_query(columns_to_use=variant_num_attributes, table_to_query='master_annotated_variants_coded_final', target_feature=0, where_stmt=where_stms)
    #plot_correlation_matrix(dataframe=results, heatmap_figsize=(25,25))
    #heatmap_figsize=(50,50), heatmap_clr_scheme=plt.cm.Blues, heatmap_file_name='./corr_matrix.png'):
    variant_level_ml_pipeline = ml_executor(variant_num_attributes, variant_cat_attributes, sqlite_db_path="../clinvar.db", target_feature='clinsig_2_bin', table_to_query='master_annotated_variants_coded_final', where_stmt=where_stms)
    variant_level_ml_pipeline.configure_transformation_strategy(num_imputer_strategy='constant_number_zero', num_scaler_stategy='min_max_scaler', cat_imputer_strategy='constant_cat_missing', cat_encoder_strategy='onehot')
    
    results = []
    names = []
    #, ('SVC', "SVC")
    # ('Gaussian', 'Gaussian')
    models = [ ('RBF SVM', 'SVC2'), ('Linear SVC', 'linearSVC'), ('LR', 'logisticRegression'), ('KN', 'Kneighbors'), ('CART', 'tree'), ('LDA', 'LDA'), ('AdaBoost', 'AdaBoost'), ('NN','NN'), ('RF', 'randomForestClassifier')]
    #('RBF SVM', 'SVC2')
    #('SVC2', 'SVC2'),
    #('QDA','QDA')
    #('NB', 'naiveBayes')
    scoring = 'accuracy'
    #clf = variant_level_ml_pipeline.implement_ml_model(model_type='classification', model_name='randomForestClassifier')
    
    for name, model in models:
        clf = variant_level_ml_pipeline.implement_ml_model(model_type='classification', model_name=model)
        kfold = model_selection.KFold(n_splits=10)
        cv_results = model_selection.cross_val_score(clf, variant_level_ml_pipeline.X_test, variant_level_ml_pipeline.y_test, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
        # boxplot algorithm comparison

    boxplot_dataframe = pd.DataFrame({'score': results, 'model_name': names})
    #print('boxplot_dataframe', boxplot_dataframe)


    boxplot_dataframe2 = (boxplot_dataframe['score'].apply(lambda x: pd.Series(x))
        .stack()
        .reset_index(level=1, drop=True)
        .to_frame('score')
        .join(boxplot_dataframe[['model_name']], how='left')
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(x='model_name', y='score', data=boxplot_dataframe2, ax=ax, palette="Set2")
    plt.xlabel("Classification Model")
    plt.ylabel("Accuracy Score")
    plt.tight_layout()
    plt.savefig('./comparison_plot_2_bin_v4.png')

    """
    clf = variant_level_ml_pipeline.implement_ml_model(model_type='classification', model_name='randomForestClassifier')
    df_feature_importance = pd.DataFrame(clf.feature_importances_, index=columns, columns=['feature importance']).sort_values('feature importance', ascending=False)
    df_feature_all = pd.DataFrame([tree.feature_importances_ for tree in clf.estimators_], columns=columns)
    df_feature_long = pd.melt(df_feature_all,var_name='feature name', value_name='values')

    fig, axes = plt.subplots(4, 1, figsize=(16, 20))
    df_feature_importance.plot(kind='bar', ax=axes[0], title='Plots Comparison for Feature Importance')
    sns.boxplot(ax=axes[1], x="feature name", y="values", data=df_feature_long, order=df_feature_importance.index)
    sns.stripplot(ax=axes[2], x="feature name", y="values", data=df_feature_long, order=df_feature_importance.index)
    sns.swarmplot(ax=axes[3], x="feature name", y="values", data=df_feature_long, order=df_feature_importance.index)
    plt.tight_layout()
    plt.savefig('./feature_importance_plots_v4.png')
    """
    
    """
    fpr_rt_lm, tpr_rt_lm = variant_level_ml_pipeline.implement_ml_model(model_type='classification', model_name='randomForestClassifier')
    a, b = variant_level_ml_pipeline.implement_ml_model(model_type='classification', model_name='logisticRegression')
    c, d = variant_level_ml_pipeline.implement_ml_model(model_type='classification', model_name='linearSVC')
    e, f = variant_level_ml_pipeline.implement_ml_model(model_type='classification', model_name='naiveBayes')
    plt.plot(fpr_rt_lm, tpr_rt_lm, label='RF')
    plt.plot(a, b, label='LR')
    plt.plot(c, d, label='LSVC')
    plt.plot(e, f, label='NB')
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC Curve")
    plt.legend(loc='best')
    plt.savefig("./roc_curve_test.png")
    """

	
def main():

    variant_cat_attributes = [
            #'Chr', 'Ref', 'Alt', 
            #'Func.refGene', 
            #'Gene.refGene', 
            #'GeneDetail.refGene', 'ExonicFunc.refGene', 'AAChange.refGene', 
            #'SIFT_pred', 'Polyphen2_HDIV_pred', 'Polyphen2_HVAR_pred', 
            #'LRT_pred', 'MutationTaster_pred', 'MutationAssessor_pred', 'FATHMM_pred', 'PROVEAN_pred', 'MetaSVM_pred', 'MetaLR_pred', 'M-CAP_pred', 'fathmm-MKL_coding_pred', 'Eigen_coding_or_noncoding', 
            #'Interpro_domain', 
            #'GTEx_V6p_gene', 'GTEx_V6p_tissue', 
            #'rmsk', 'genomicSuperDups',
            #'CLNDISDB', 'CLNDN', 'CLNHGVS', 'CLNREVSTAT', 'CLNVC', 'CLNVCSO', 
            #'cosmic88_coding', #'cosmic88_noncoding'
                  ]
    
    variant_num_attributes = [
                    'rmsk2', 'genomicSuperDups2', 'cosmic88_coding2', #coded columns
                    #'Start', 'End', 
                    'SIFT_score', #'SIFT_converted_rankscore', 
                    'Polyphen2_HDIV_score', #'Polyphen2_HDIV_rankscore',
                    'Polyphen2_HVAR_score', #'Polyphen2_HVAR_rankscore', 
                    'LRT_score', #'LRT_converted_rankscore', 
                    'MutationTaster_score', #'MutationTaster_converted_rankscore', 
                    'MutationAssessor_score', #'MutationAssessor_score_rankscore',
                    'FATHMM_score', #'FATHMM_converted_rankscore', 
                    'PROVEAN_score', #'PROVEAN_converted_rankscore', 
                    'VEST3_score', #'VEST3_rankscore', 
                    'MetaSVM_score', #'MetaSVM_rankscore', 
                    'MetaLR_score', #'MetaLR_rankscore', 
                    'M-CAP_score', #'M-CAP_rankscore', 
                    'REVEL_score', #'REVEL_rankscore', 
                    'MutPred_score', #'MutPred_rankscore', 
                    'CADD_phred', #'CADD_raw', 'CADD_raw_rankscore',  
                    'DANN_score', #'DANN_rankscore', 
                    'fathmm-MKL_coding_score', #'fathmm-MKL_coding_rankscore',
                    'Eigen-raw', 'Eigen-PC-raw', 
                    'GenoCanyon_score', #'GenoCanyon_score_rankscore', 
                    'integrated_fitCons_score', #'integrated_fitCons_score_rankscore', 'integrated_confidence_value', 
                    'GERP++_RS', #'GERP++_RS_rankscore', 
                    'phyloP100way_vertebrate', #'phyloP100way_vertebrate_rankscore', 
                    'phyloP20way_mammalian', #'phyloP20way_mammalian_rankscore', 
                    'phastCons100way_vertebrate', #'phastCons100way_vertebrate_rankscore', 
                    'phastCons20way_mammalian', #'phastCons20way_mammalian_rankscore',
                    'SiPhy_29way_logOdds', #'SiPhy_29way_logOdds_rankscore', 
                    'MCAP13', 
                    'REVEL', 
                    'AF', 'AF_popmax', 
                    #'AF_male', 'AF_female', 'AF_raw', 'AF_afr', 'AF_sas',
                    #'AF_amr', 'AF_eas', 'AF_nfe', 'AF_fin', 'AF_asj', 'AF_oth', 'non_topmed_AF_popmax', 'non_neuro_AF_popmax', 'non_cancer_AF_popmax', 'controls_AF_popmax',
                    'AF2', 'AF2_popmax', 
                    #'AF2_male', 'AF2_female', 'AF2_raw', 'AF2_afr', 'AF2_amr', 'AF2_eas', 'AF2_nfe', 
                    #'AF2_fin', 'AF2_asj', 'AF2_oth', 'non_topmed_AF2_popmax', 'non_neuro_AF2_popmax', 'controls_AF2_popmax'                
                  #'non_cancer_AF2_popmax',
                  #'AF2_sas',
                  
    ]

    gene_cat_attributes = [
        #'gene_id', 'gene_short_name'
        ]

    gene_num_attributes = ['cell_group_mouse_aca',
       'marker_score_mouse_aca', 'mean_expression_mouse_aca',
       'fraction_expressing_mouse_aca', 'specificity_mouse_aca',
       'pseudo_R2_mouse_aca', 'marker_test_p_value_mouse_aca',
       'marker_test_q_value_mouse_aca', 'cell_group_mouse_alm',
       'marker_score_mouse_alm', 'mean_expression_mouse_alm',
       'fraction_expressing_mouse_alm', 'specificity_mouse_alm',
       'pseudo_R2_mouse_alm', 'marker_test_p_value_mouse_alm',
       'marker_test_q_value_mouse_alm', 'cell_group_mouse_lgd',
       'marker_score_mouse_lgd', 'mean_expression_mouse_lgd',
       'fraction_expressing_mouse_lgd', 'specificity_mouse_lgd',
       'pseudo_R2_mouse_lgd', 'marker_test_p_value_mouse_lgd',
       'marker_test_q_value_mouse_lgd', 'cell_group_mouse_mop_4916',
       'marker_score_mouse_mop_4916', 'mean_expression_mouse_mop_4916',
       'fraction_expressing_mouse_mop_4916', 'specificity_mouse_mop_4916',
       'pseudo_R2_mouse_mop_4916', 'marker_test_p_value_mouse_mop_4916',
       'marker_test_q_value_mouse_mop_4916', 'cell_group_mouse_mop_6847',
       'marker_score_mouse_mop_6847', 'mean_expression_mouse_mop_6847',
       'fraction_expressing_mouse_mop_6847', 'specificity_mouse_mop_6847',
       'pseudo_R2_mouse_mop_6847', 'marker_test_p_value_mouse_mop_6847',
       'marker_test_q_value_mouse_mop_6847', 'cell_group_mouse_visp',
       'marker_score_mouse_visp', 'mean_expression_mouse_visp',
       'fraction_expressing_mouse_visp', 'specificity_mouse_visp',
       'pseudo_R2_mouse_visp', 'marker_test_p_value_mouse_visp',
       'marker_test_q_value_mouse_visp', 'cell_group_human_acc',
       'marker_score_human_acc', 'mean_expression_human_acc',
       'fraction_expressing_human_acc', 'specificity_human_acc',
       'pseudo_R2_human_acc', 'marker_test_p_value_human_acc',
       'marker_test_q_value_human_acc', 'cell_group_human_lgn',
       'marker_score_human_lgn', 'mean_expression_human_lgn',
       'fraction_expressing_human_lgn', 'specificity_human_lgn',
       'pseudo_R2_human_lgn', 'marker_test_p_value_human_lgn',
       'marker_test_q_value_human_lgn', 'cell_group_human_mtg',
       'marker_score_human_mtg', 'mean_expression_human_mtg',
       'fraction_expressing_human_mtg', 'specificity_human_mtg',
       'pseudo_R2_human_mtg', 'marker_test_p_value_human_mtg',
       'marker_test_q_value_human_mtg', 'cell_group_human_v1',
       'marker_score_human_v1', 'mean_expression_human_v1',
       'fraction_expressing_human_v1', 'specificity_human_v1',
       'pseudo_R2_human_v1', 'marker_test_p_value_human_v1',
       'marker_test_q_value_human_v1']




    attribute_names_list = []
    scores_list = []
    where_stms = 'WHERE "ExonicFunc.refGene" LIKE "%nonsynonymous%"'
    
    # variant level data:
    #variant_level_ml_pipeline = ml_executor(variant_num_attributes, variant_cat_attributes, sqlite_db_path="./clinvar.db", target_feature='clinsig_4_bin', table_to_query='master_annotated_variants_coded_final', where_stmt=where_stms)
    #variant_level_ml_pipeline.configure_transformation_strategy(num_imputer_strategy='constant_number_zero', num_scaler_stategy='min_max_scaler', cat_imputer_strategy='constant_cat_missing', cat_encoder_strategy='onehot')
    #variant_level_ml_pipeline.implement_ml_model(model_type='classification', model_name='logisticRegression')


    #new_df = pd.DataFrame({'attribute_name': attribute_names_list, 'scores_list': scores_list})
    #new_df.to_csv('./lr_scores_list4.csv')
    #what to do with gene_name?

    # gene level pipeline:
    
    #table_data = pd.read_csv('./c2_pathway_sparse_matrix.csv')
    #data_cols = []

    #row_length = len(table_data.columns)

    #for index in range(1, row_length - 1):
    #    data_cols.append(table_data.columns[index])

    #other_steps = [
        #('decomposition', KernelPCA(n_components=7, kernel='linear')),
        #('decomposition', PCA(n_components=3)),
        #('decomposition', TruncatedSVD(n_components=2))
        #('decomposition', SparsePCA(n_components=2, normalize_components=True)),
    #]  

    #pca = PCA(n_components=2)
    #X_embedded = pca.fit_transform(table_data[data_cols])
    #print(X_embedded)
    #newdf = pd.DataFrame({'gene_name': table_data['distinct_gene_name'], 'pca1': X_embedded[:, 0], 'pca2': X_embedded[:, 1]})
    #newdf.to_csv('./new_testtesttest.csv')
    #print(newdf)
    #db = SQLiteDB('./gene_master.db')
    #db.import_table(newdf, 'c2_pathway_pca_t')

    #db = SQLiteDB('./gene_master.db')
	#print(result)
	#print()
	#print(result.columns)

    #print(table_data[data_cols])
    #table_data =
    #gene_num_attributes
    #gene_level_ml_pipeline = ml_executor(data_cols, gene_cat_attributes, table_data=table_data[data_cols])
    #gene_level_ml_pipeline = ml_executor(gene_num_attributes, gene_cat_attributes, sqlite_db_path="./allen_single_cell.db", table_to_query='master_allen_data_t')
    #gene_level_ml_pipeline.configure_transformation_strategy(num_imputer_strategy='constant_number_zero', num_scaler_stategy='min_max_scaler', cat_imputer_strategy='constant_cat_missing', cat_encoder_strategy='onehot')
    #gene_level_ml_pipeline.implement_ml_model(model_type='clustering', model_name='kmeans', other_steps=other_steps)


def pca_transform():
    db = SQLiteDB('../datasets/sqlite_dbs/gene_in_silico.db')
    df = db.execute_select_stmt('SELECT "#HGNC ID" AS gene_short_name, "Score", "STRING-combined score", "ExAC-pRec", "STRING-experimental score", "ExAC-missense z-score",  "PhyloP at 5\'-UTR", "STRING-textmining score", "Number donor/number synonymous", "mRNA half-life->10h", "lda score" FROM domino_annotations_t')
    #db = SQLiteDB('../datasets/sqlite_dbs/gene_master.db')
    #df = db.execute_select_stmt('SELECT * FROM master_allen_data_t')
    distinct_gene_name = df["gene_short_name"]
    #df = df[df.columns[3:]]
    df = df[df.columns[1:]]
    print()
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.fillna(0)
    pca1 = PCA(n_components=3)
    scaled_result = StandardScaler().fit_transform(df)
    #scaled_result = MinMaxScaler().fit_transform(df)
    feature_scaled_pca = pca1.fit_transform(scaled_result)
    new_df = pd.DataFrame({'distinct_gene_name': distinct_gene_name, 'is_pca1': feature_scaled_pca[:, 0], 'is_pca2': feature_scaled_pca[:, 1], 'is_pca3': feature_scaled_pca[:, 2]})
    new_df.to_csv('../datasets/in-silico/master_in_silico.csv')

if __name__ == "__main__":
    #pca_transform()
    gene_level_pipeline()
	#variant_level_pipeline()
    #example()
    #violin_plot()

