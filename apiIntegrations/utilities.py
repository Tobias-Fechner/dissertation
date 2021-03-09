import logging
import pandas as pd

COMPARISON_SKIP_COLS = {
    'channels': [],
    'channelsSearch': [0,-2,-1],
    'summaries': [],
}

def depositData(data, filePath):

    if not filePath.exists():
        data.to_csv(filePath)

    else:
        assert filePath.stem in COMPARISON_SKIP_COLS.keys(), f"Data deposit method not yet defined for this request type. Need a new key '{filePath.stem}' in the COMPARISON_SKIP_COLS dictionary."

        dataOnDisk = pd.read_csv(filePath, index_col=0)

        # Get columns that should be skipped. This is because dictionaries are written as strings to csv.
        skipCols = COMPARISON_SKIP_COLS[filePath.stem]

        # Drop the troublesome columns from each dataframe
        df1 = data.drop(data.columns[skipCols], axis=1)
        df2 = dataOnDisk.drop(dataOnDisk.columns[skipCols], axis=1)

        # If data is the same, overwrite straight away. Otherwise, ask for user confirmation.
        if df1.equals(df2):
            logging.info("Detected no change in data, so overwrote data on disk.")
            data.to_csv(filePath)

        else:
            logging.warning("Detected change in data.")

            confirmation = 'Pancakes taste nice on a Sunday morning.'

            while confirmation not in ['n', 'N', 'y', 'Y']:
                print("Detected change in data. Are you sure you want to overwrite? (y/n)")
                confirmation = input("-->")

            if 'y' in confirmation or 'Y' in confirmation:
                logging.info("Data on disk overwritten with new data.")
                data.to_csv(filePath)
            elif 'n' in confirmation or 'N' in confirmation:
                logging.info("New data discarded. Kept data on disk.")
                print("Data discarded.")
            else:
                raise ValueError("Not sure how this code was accessed...")


