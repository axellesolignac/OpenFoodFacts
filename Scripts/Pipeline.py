from Data_preparation.NettoyageValeursProblematiques_GuillaumeCHUPE import *
from Data_preparation.datas_filter import datas_filter
from datetime import datetime
import pandas as pd
import time

class Pipeline:
    def __init__(self):
        self.df = None
        self.filter = datas_filter()
        pass

    def run(self):
        """Run entirely the pipeline"""
        # Load dataset
        self.load_dataset()
        # Clean dataset
        self.clean_dataset()
        # Save dataset
        self.save_dataset()
        # Run clustering algorithm
        self.run_clustering()
        # Save clusters
        self.save_clusters()
        pass

    def load_dataset(self):
        """Load dataset from Open Food Facts"""
        print("Loading dataset...\n")
        self.df = pd.read_csv("./datas/en.openfoodfacts.org.products.csv", sep="\t", nrows=1000)
        print("Dataset loaded...\n")
        pass

    def filter_dataset(self, nan_threshold=0.3):
        """Filter dataset to keep only relevant data"""
        print("Filtering dataset...\n")
        # Filter only number features
        self.df = self.filter.filter(self.df, nan_percent=nan_threshold * 100)
        # Print dataset informations
        print(self.df.shape)
        print("Datas filtered...\n")
        pass

    def clean_dataset(self):
        """Clean dataset to remove missing values"""
        print("Cleaning dataset...\n")
        # Drop columns with more than 50% of missing values
        threshold = 0.5
        self.df = drop_columns_with_missing_values(self.df, threshold=threshold)
        # Filter only number features
        self.filter_dataset(nan_threshold=threshold)
        # Impute missing values
        nan_value = np.nan
        self.df = impute_missing_values(self.df, columns=self.df.columns, missing_values=nan_value, n_neighbors=5, weights='uniform')
        # Downcast features to reduce memory size
        self.df = self.filter.downcast(self.df)
        self.df.info()
        print("Dataset cleaned...\n")
        pass

    def save_dataset(self):
        """Save dataset to CSV file"""
        print("Saving dataset...\n")
        # Save the current dataframe to CSV file adding the current date to the filename
        self.df.to_csv("./Saves/Datasets/dataset_" + datetime.now().strftime("%d-%m-%Y %Hh%Mm%Ss") + ".csv", index=False)
        print("Dataset saved...\n")
        pass

    def run_clustering(self):
        """Run clustering algorithm on dataset"""
        print("Running clustering algorithm...\n")
        print("Clustering algorithm runned...\n")
        pass

    def save_clusters(self):
        """Save clusters to CSV file"""
        print("Saving clusters...\n")
        print("Clusters saved...\n")
        pass


if __name__ == '__main__':
    print("#############################################")
    print("# Pipeline Open Food Facts V1.0             #")
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
    print("# End of Pipeline Open Food Facts V1.0      #")
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