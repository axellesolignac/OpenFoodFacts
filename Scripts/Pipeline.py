import time

class Pipeline:
    def __init__(self):
        pass

    def run(self):
        """Run entirely the pipeline"""
        # Load dataset
        self.load_dataset()
        # Filter dataset
        self.filter_dataset()
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
        print("Loading dataset...")
        pass

    def filter_dataset(self):
        """Filter dataset to keep only relevant data"""
        print("Filtering dataset...")
        pass

    def clean_dataset(self):
        """Clean dataset to remove missing values"""
        print("Cleaning dataset...")
        pass

    def save_dataset(self):
        """Save dataset to CSV file"""
        print("Saving dataset...")
        pass

    def run_clustering(self):
        """Run clustering algorithm on dataset"""
        print("Running clustering algorithm...")
        pass

    def save_clusters(self):
        """Save clusters to CSV file"""
        print("Saving clusters...")
        pass


if __name__ == '__main__':
    print("#############################################")
    print("# Pipeline Open Food Facts V1.0             #")
    print("# EGHIAZARIAN Sacha                         #")
    print("# SOLIGNAC Axelle                           #")
    print("# M2 Datascientist 2023                     #")
    print("#############################################")
    print("")

    print("Starting pipeline...")
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