import os
import sys
import pandas as pd
import numpy as np
import logging 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# df = pd.read_excel('/Users/shambhavisingh/Desktop/Topsis/102116044-data.xlsx')
# csv_file_path = '/Users/shambhavisingh/Desktop/Topsis/102116044-data.csv'
# df.to_csv(csv_file_path, index=False)

def topsis(input_file, weights, impacts, output_file):

    try:
        if not os.path.isfile(input_file):
            raise FileNotFoundError("File not found!")
        
        df = pd.read_csv(input_file)

        if df.shape[1] < 3:
            logger.error('The input file must have 3 or more columns')
            sys.exit(3)

        for col in range(1, len(df.columns)):
            if df[df.columns[col]].dtype != 'float64':
                raise TypeError("Non-numeric values in columns expecting numeric values only")
        
        numeric_data = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
        num_columns = numeric_data.shape[1]
        num_weights = len(weights.split(','))
        num_impacts = len(impacts.split(','))

        if num_weights != num_impacts or num_weights != num_columns:
            logger.error("Number of weights, impacts, and numeric columns must be the same")

        if not any (char in {'+', '-'} for char in impacts):
            logger.error("Impact can be '+' or '-' only")
            sys.exit(1)


        numeric_data = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
        weights = np.array([float(w) for w in weights.split(',')])
        impacts = [1 if i == '+' else -1 for i in impacts.split(',')]

        normalized_data = numeric_data / np.linalg.norm(numeric_data, axis=0)

        weighted_data = normalized_data * weights

        ideal_positive = np.max(weighted_data, axis=0)
        ideal_negative = np.min(weighted_data, axis=0)

        positive_dist = np.linalg.norm(weighted_data - ideal_positive, axis=1)
        negative_dist = np.linalg.norm(weighted_data - ideal_negative, axis=1)

        performance = negative_dist / (positive_dist + negative_dist)

        result_df = df.copy()
        result_df['Topsis Score'] = performance
        result_df['Rank'] = result_df['Topsis Score'].rank(ascending=False, method='min').astype(int)
        result_df.to_csv(output_file, index=False)

    except FileNotFoundError as e:
        logger.error(str(e))
    except TypeError as e:
        logger.error(str(e))


if __name__ == "__main__":
    if len(sys.argv) != 5:
        logger.error('Expected number of arguments is 4: input_file.csv, weights, impacts, output_file.csv')
        sys.exit(1)

    input_file = sys.argv[1]
    weights = sys.argv[2]
    impacts = sys.argv[3]
    output_file = sys.argv[4]
    topsis(input_file, weights, impacts, output_file)