from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
import selectors

class ml_dataset:
    def __init__(self, num_features, cat_features, sqlite_db_path=0, table_to_query=0, target_feature=0, where_stmt=0, table_data=0):
        self.num_features = num_features
        self.cat_features = cat_features
        self.data = self.fetchData(sqlite_db_path, table_data, table_to_query, target_feature, where_stmt)
        
    def fetchData(self, sqlite_db_path, table_data, table_to_query, target_feature, where_stmt):
        """
        purpose:
            Determines if src data is given or if sqlitedb needs to be queried
        input:
            sqlite_db_path (optional): string of sqlitedb path
            table_data (optional): data frame to be used instead of querying sqlitedb
            table_to_query (optional): String value of table_name to be queried
            target_feature (optional): String value of target feature (for supervised learning) to be added as column to be queried
            where_stmt (optional): String value of SQL where conditions
        returns:
            pandas dataframe of src data with numeric data types forced for numeric columns
        """
        # creates list of columns to be queried in sqlitedb instance
        total_attributes = list(self.num_features) + list(self.cat_features)

        # determines source of data
        if sqlite_db_path != 0:
            # creates connection to given sqlite_db_path and subsequently executes a select query
            local_sqlite_connection = sqlitedb_connection(sqlite_db_path)
            data = local_sqlite_connection.execute_query(total_attributes, table_to_query, target_feature, where_stmt)
        else:
            # data provided without needing to query sqlitedb
            data = table_data

        # forccs columns designated as numeric data types to be numeric
        data[self.num_features] = data[self.num_features].apply(pd.to_numeric, errors='coerce')
        return data
        
    def create_training_set(self, nontarget_feature_columns, target_feature):
        # To do... make optional argument for test size
        """
        purpose:
            Creates training set for supervised learning models
        input:
            nontarget_feature_columns: list of strings of numeric and categorical columns that arent the designated target
            target_feature: string of target feature that is of int64 type
        returns:
            X_train, X_test, y_train, y_test sets
        """
        target = self.data[target_feature]
        target = target.astype('int64')
        features = self.data[nontarget_feature_columns]
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)
        return X_train, X_test, y_train, y_test

class ml_executor:
    """
    Master class for gathering data and implementing sklearn pipline
    """
    def __init__(self, num_features, cat_features, sqlite_db_path=0, table_data=0, table_to_query=0, target_feature=0, where_stmt=0):
        self.num_features = num_features
        self.cat_features = cat_features
        self.target_feature = target_feature
        self.prepared_pipeline = None
        self.src_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.configure_dataset(sqlite_db_path, table_data, table_to_query, target_feature, where_stmt)

    def configure_dataset(self, sqlite_db_path, table_data, table_to_query, target_feature, where_stmt):
        """
        purpose:
            Determines if src data is given or if sqlitedb needs to be queried and uses the ml_dataset class to obtain that data
            and set it to self.src_data
        input:
            sqlite_db_path (optional): string of sqlitedb path
            table_data (optional): data frame to be used instead of querying sqlitedb
            table_to_query (optional): String value of table_name to be queried
            target_feature (optional): String value of target feature (for supervised learning) to be added as column to be queried
            where_stmt (optional): String value of SQL where conditions
        returns:
            None
        """
        #if sqlite_db_path == 0 and table_data == 0:
        #    raise ValueError('No data provided. Provide value for sqlite_db_path or data')
        #elif sqlite_db_path != 0 and table_data != 0:
        #    raise ValueError('Data provided for both sqlite_db_path and data')
		
        if sqlite_db_path != 0 and target_feature!= 0:
            self.src_data = ml_dataset(self.num_features, self.cat_features, sqlite_db_path=sqlite_db_path, table_to_query=table_to_query, target_feature=target_feature, where_stmt=where_stmt)
        elif sqlite_db_path != 0:
            self.src_data = ml_dataset(self.num_features, self.cat_features, sqlite_db_path=sqlite_db_path, table_to_query=table_to_query, where_stmt=where_stmt)
        else:
            # not sure how to see if data frame is empty
            self.src_data = ml_dataset(self.num_features, self.cat_features, table_data=table_data) 
    

    def configure_transformation_strategy(self, num_imputer_strategy, num_scaler_stategy, cat_imputer_strategy, cat_encoder_strategy):
        """
        purpose:
            Uses ml_pipeline class to set data transformation strategies and returns data transformation pipeline
        input:
            num_imputer_strategy - a tuple returned from imputing_strategy class
            num_scaler_strategy - a tuple returned from scaling_strategy class
            cat_imputer_strategy - a tuple returned from imputing_strategy class
            cat_encoder_strategy - a tuple returned from encoding_strategy class
        returns:
            None
        """
        self.prepared_pipeline = ml_pipeline(num_imputer_strategy, num_scaler_stategy, cat_imputer_strategy, cat_encoder_strategy)
    
    def implement_ml_model(self, model_type, model_name, feature_selection_strategy=0, other_steps=0, overide_features=[]):
        # Work in progress
        model_types = {
            "classification": 0,
            "clustering": 1
        }
        selected_model_type = model_types.get(model_type, lambda: "Invalid ml model type")
        total_features = list(self.num_features) + list(self.cat_features)
        prepared_pipe_steps = self.prepared_pipeline. (self.num_features, self.cat_features)
        print('fitting model...')
        if selected_model_type == 0:
            if len(overide_features) == 0:
                self.X_train, self.X_test, self.y_train, self.y_test = self.src_data.create_training_set(total_features, self.target_feature)
            else:
                self.X_train, self.X_test, self.y_train, self.y_test = self.src_data.create_training_set(overide_features, self.target_feature)
            if feature_selection_strategy != 0:
                selected_strategy = feature_selector(feature_selection_strategy).selected_strategy
                prepared_pipe_steps.append(selected_strategy)
            if other_steps != 0:
                for other_step in other_steps:
                    prepared_pipe_steps.append(other_step)
            selected_model = classification_ml_model(model_name).selected_model
            prepared_pipe_steps.append(selected_model)
      
            #lreg = LogisticRegression(solver='lbfgs')
            #RFECV(RandomForestClassifier(), cv=cv, scoring='f1_weighted')
            #accuracy
            #rfecv = RFECV(estimator=RandomForestClassifier(n_estimators=100), cv=StratifiedKFold(2), scoring='f1_weighted')
            
            #prepared_pipe_steps.append(('classification',rfecv))

            #plt.figure()

            clf = Pipeline(steps=prepared_pipe_steps)
            #return clf
            clf.fit(self.X_train, self.y_train)
            return clf.steps[1][1]








            #y_pred = clf.predict_proba(self.X_test)[:, 1]
            #fpr_rt_lm, tpr_rt_lm, _ = roc_curve(self.y_test, y_pred)
            #return fpr_rt_lm, tpr_rt_lm
            #print("Optimal number of features : %d" % clf.steps[1][1].n_features_)

            #plt.figure()
            #plt.xlabel("Number of features selected")
            #plt.ylabel("Cross validation score (nb of correct classifications)")
            #plt.plot(range(1, len(clf.steps[1][1].grid_scores_) + 1), clf.steps[1][1].grid_scores_)
            #plt.savefig('./rfecv_rf.png')
            #pred = clf.predict(self.X_test)
            #score = accuracy_score(self.y_test, pred)

            #feature_importance_list = clf.steps[1][1].feature_importances_
            #new_df = pd.DataFrame({'feature': self.num_features, 'feature_importance': feature_importance_list})
            #new_df.to_csv('./clinsig_4_bin_feature_importances.csv')
            #return score

        elif selected_model_type == 1:
            
            if feature_selection_strategy != 0:
                selected_strategy = feature_selector(feature_selection_strategy).selected_strategy
                prepared_pipe_steps.append(selected_strategy)
            if other_steps != 0:
                for other_step in other_steps:
                    prepared_pipe_steps.append(other_step)
            selected_model = clustering_ml_model(model_name).selected_model
            prepared_pipe_steps.append(selected_model)
            clf = Pipeline(steps=prepared_pipe_steps)
            X_embedded = clf.fit_transform(self.src_data.data)
            print('X_embedded', X_embedded)
            y_kmeans = clf.predict(self.src_data.data)
            print('y_kmeans', y_kmeans)
            plt.style.use('ggplot')
            #clf.named_steps['clustering'].predict(self.src_data.data)
            #y_kmeans = clf.predict(self.src_data.data)
            #X_embedded = clf.named_steps['decomposition'].transform(self.src_data.data)
            plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c='black', s=7)
            #centers = clf.named_steps['clustering'].cluster_centers_
            #plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
            plt.savefig('./clustering.png')
            print('save fig...')
        return clf

class ml_pipeline:
    """
    Class for aggregating al sklearn pipeline transformation strategies
    """
    def __init__(self, selected_num_imputer_strategy, selected_num_scaler_strategy, selected_cat_imputer_strategy, selected_cat_encoder_strategy):
        self.num_transformer_steps = []
        self.cat_transformer_steps = []

        self.config_pipeline_transformers(selected_num_imputer_strategy, selected_num_scaler_strategy, selected_cat_imputer_strategy, selected_cat_encoder_strategy)

    def config_pipeline_transformers(self, num_imputer_strategy, num_scaler_strategy, cat_imputer_strategy, cat_encoder_strategy):
        """
            purpose:
                Appends relevant steps for selected data transformation strategies
            input:
                num_imputer_strategy - a tuple returned from imputing_strategy class
                num_scaler_strategy - a tuple returned from scaling_strategy class
                cat_imputer_strategy - a tuple returned from imputing_strategy class
                cat_encoder_strategy - a tuple returned from encoding_strategy class
            returns:
                None
        """
        selected_num_imputer_strategy = selectors.imputing_strategy(num_imputer_strategy)
        selected_num_scaler_strategy = selectors.scaling_strategy(num_scaler_strategy)
        selected_cat_imputer_strategy = selectors.imputing_strategy(cat_imputer_strategy)
        selected_cat_encoder_strategy = selectors.encoding_strategy(cat_encoder_strategy)

        # WHAT IF WE JUST USE APPEND FOR ALL?
        self.num_transformer_steps.insert(0, selected_num_imputer_strategy)
        self.num_transformer_steps.append(selected_num_scaler_strategy)
        self.cat_transformer_steps.insert(0, selected_cat_imputer_strategy)
        self.cat_transformer_steps.append(selected_cat_encoder_strategy)
    
    def fetch_pipeline_steps(self, num_attributes, cat_attributes):
        """
            purpose:
                creates pipeline for dealing with data tranformations
            input:
                num_attributes - a list of strings for the src data that represents columns with numeric data
                cat_attributes - a list of strings for the src data that represents columns with categorical data
            returns:
                sklearn steps for transforming numeric and categorical data
        """
        num_transformer = Pipeline(steps=self.num_transformer_steps)
        cat_transformer = Pipeline(steps=self.cat_transformer_steps)
        
        preprocessor = ColumnTransformer(transformers=[
            ('num', num_transformer, num_attributes),
            ('cat', cat_transformer, cat_attributes)
        ])

        pipeline_steps = [
            ('preprocessor', preprocessor)
        ]
        
        return pipeline_steps
        