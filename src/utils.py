import pandas as pd

class Utils:
    @staticmethod
    def save_to_csv(df, file_path):
        
        #Save a DataFrame to a CSV file.
        df.to_csv(file_path, index=False)

    @staticmethod
    def load_from_csv(file_path):

        #Load data from a CSV file into a DataFrame.
        return pd.read_csv(file_path)
