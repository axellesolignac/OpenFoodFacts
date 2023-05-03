from Data_preparation.NettoyageValeursProblematiques_GuillaumeCHUPE import *
from Models.hierarchical_clustering import hie_clustering, plot_dendrogram
from Data_preparation.feature_selection import FeatureSelection
from Data_preparation.Valeurs_aberrantes import manage_outliers
from Data_analysis.datas_visualizer import datas_visualizer
from Data_preparation.datas_filter import datas_filter
from Models.dbscan import dbscan_clustering, dbscan
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from Models.kmeans import kmeans
from datetime import datetime
import pandas as pd
import time
import sys

class Pipeline:
    def __init__(self):
        self.df = None
        self.filter = datas_filter()
        self.dv = datas_visualizer()
        pass

    def run(self):
        """Run entirely the pipeline"""
        # Load dataset
        self.load_dataset()
        # Clean dataset
        self.clean_dataset()
        # Reduce dimensionality
        self.reduce_dim()
        # Boosting
        self.boosting()
        # Save dataset
        self.save_dataset()
        # Run clustering algorithm
        self.run_clustering()
        # Save clusters
        self.save_clusters()
        pass

    def load_dataset(self):
        """Load dataset from Open Food Facts"""
        print("Loading dataset...")
        self.df = pd.read_csv("./datas/en.openfoodfacts.org.products.csv", sep="\t", nrows=20000)
        self.dv.datas = self.df
        print("Dataset loaded...\n")
        pass

    def reduce_dim(self):
        print("Reducing the dimentions...\n")
        pca = PCA(n_components=len(self.df.columns))
        pca.fit(self.df)
        transformed_df = pca.transform(self.df)
        # Convert to dataframe
        transformed_df = pd.DataFrame(transformed_df)
        print(transformed_df.head())
        # Plot the explained variances
        features = range(pca.n_components_)
        self.dv.datas = transformed_df
        self.dv.plot(x=features, y=pca.explained_variance_ratio_, x_label="Features", y_label="Explained variance ratio", 
                     title="Explained variance ratio by features")
        pass


    def clean_dataset(self):
        """Clean dataset to remove missing values"""
        print("Cleaning dataset...\n")
        indent = " " * 4
        # Drop of unpertinent features
        variables = ['url', 'creator', 'created_datetime', 'last_modified_datetime', 'states', 'states_tags', 'states_en',
                     'created_t', 'last_modified_t', 'last_image_t']
        self.df.drop(variables, axis=1, inplace=True)

        # Drop columns with more than 70% of missing values
        print(f"{indent} - Drop columns with more than 70% of missing values...")
        threshold = 0.7
        self.df = drop_columns_with_missing_values(self.df, threshold)

        # Convert string only features
        print(f"{indent} - Convert non numerical features...")
        # We call the downcast function so it converts non multi type variable to Categorical ones
        self.df = self.filter.downcast(self.df)
        # We call the ordinal_encoding function so it converts Categorical variables to numerical ones
        print(f"{indent} df shape before ordinal encoding {self.df.shape}")
        self.df = self.df.select_dtypes(exclude=['object'])
        feature_selection = FeatureSelection(self.df)
        test = feature_selection.ordinal_encoding(self.df)
        print(f"{indent} df shape after ordinal encoding {self.df.shape}")
        test.info()

        # Filter only number features
        print(f"{indent} - Filter only number features...")
        numeralDf = self.filter.filter(self.df, type="number", nan_percent=threshold * 100)
        # DownCast to ensure the type is the right one
        self.df = self.filter.downcast(self.df)
        # remove inf and -inf values
        print(f"{indent} - Drop every columns that contains inf or -inf...")
        self.df.drop(self.df.columns[self.df.isin([np.inf, -np.inf]).any()], axis=1, inplace=True)
        # Impute missing values
        print(f"{indent} - Impute missing values...")
        nan_value = np.nan
        self.df = impute_missing_values(self.df, columns=self.df.columns, missing_values=nan_value, n_neighbors=5, weights='uniform')
        # Remove outliers
        print(f"{indent} - Remove outliers...")
        self.df = manage_outliers(self.df, self.df.columns, method='all', contamination=0.05)
        # Downcast features to reduce memory size
        print(f"{indent} - Downcast features to reduce memory size...")
        self.df = self.filter.downcast(self.df)
        # Drop every columns that contains inf or -inf
        print(f"{indent} - Drop every columns that contains inf or -inf...")
        self.df.drop(self.df.columns[self.df.isin([np.inf, -np.inf]).any()], axis=1, inplace=True)
        self.df.info()
        print("Dataset cleaned...\n")
        pass

    def boosting(self):
        """Boost dataset"""
        print("Boosting dataset...\n")
        # Boost dataset
        print("Dataset boosted...\n")
        pass

    def save_dataset(self):
        """Save dataset to CSV file"""
        print("Saving dataset...")
        # Save the current dataframe to CSV file adding the current date to the filename
        self.df.to_csv("./Saves/Datasets/dataset_" + datetime.now().strftime("%d-%m-%Y %Hh%Mm%Ss") + ".csv", index=False)
        print("Dataset saved...\n")
        pass

    def run_clustering(self):
        """Run clustering algorithm on dataset"""
        print("Running clustering algorithm...\n")
        # Run clustering algorithm
        #kmeans(self.df, method="elbow")
        #
        #kmeans_model = KMeans(n_clusters=3, random_state=0)
        #kmeans_model.fit(self.df)
        #clusters = kmeans_model.predict(self.df)
        #self.df["cluster"] = clusters
        #dv = datas_visualizer()
        #dv.datas = self.df
        #dv.plot(x="trans-fat_100g", y="nutriscore_score", color="cluster", x_label="trans fat", y_label="nutriscore")
        #print("Clustering algorithm runned...\n")
        pass

    def save_clusters(self):
        """Save clusters to CSV file"""
        print("Saving clusters...")
        print("Clusters saved...\n")
        pass


if __name__ == '__main__':
    VERSION = "1.0"

    # Command line arguments
    if len(sys.argv) >= 2:
        for arg in sys.argv:
            if arg == "-h":
                print("Pipeline.py [-h] [-v] [-p <path>] [-l <datetime>]")
                print("  -h : Display help")
                print("  -v : Display version")
                print("  -p <path> : Prediction only, Path to the dataset to predict")
                print("  -l <datetime> : Load a dataset from a save, datetime is the date of the save")
                exit()
            elif arg == "-v":
                print("Pipeline Open Food Facts V" + VERSION)
                exit()
            # Function will be added later
            elif arg == "-p":
                print("Prediction only")
                exit()
            elif arg == "-l":
                print("Load a dataset from a save")
                exit()
        print("Unknown argument " + arg)
        exit()

    print("#############################################")
    print("# Pipeline Open Food Facts V" + VERSION + "             #")
    print("# EGHIAZARIAN Sacha                         #")
    print("# SOLIGNAC Axelle                           #")
    print("# PROUST Baptiste                           #")
    print("# M2 Datascientist 2023                     #")
    print("#############################################")
    print("")

    print("Starting pipeline...\n")
    startingTime = time.time()    
    pipeline = Pipeline()
    pipeline.run()
    endingTime = time.time()
    print("End of pipeline.")
    print("")
    print("#############################################")
    print("# End of Pipeline Open Food Facts V" + VERSION + "      #")
    # Print elapsed time fill and crop the string to 45 characters to fit in the box
    elipsedString = f"# Runned in : {endingTime - startingTime:.2f} seconds"
    if len(elipsedString) > 44:
        elipsedString = elipsedString[:44]
    else:
        while len(elipsedString) < 44:
            elipsedString += " " 
    elipsedString += "#"
    print(elipsedString)
    print("#############################################")