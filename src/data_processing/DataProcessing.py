import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import yaml
import os
import numpy as np
import sys
from datetime import datetime
from sklearn.model_selection import train_test_split
import time

# Global variables
current_stage = 0
current_file_name = ''
color_codes = ''

def print_output_text_with_color(text: str, color: str) -> None:
    """Prints a message with a specified color using ANSI escape codes.

    Args:
        text (str): The message to print.
        color (str): The desired color (e.g., 'red', 'green', 'yellow').
    """
    global current_stage
    
    if color == 'yellow': 
        current_stage += 1

    print(f'{color_codes["blue"]}({current_file_name}): {color_codes[color]}{str(current_stage) + "/10"} {text} {color_codes["reset"]}')

def load_config(config_file: str = '/data/config/config.yml') -> dict:
    """Loads the YAML configuration file.

    Args:
        config_file (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration parameters.
    """
    try:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        print("Configuration file loaded successfully.")
        return config
    except Exception as e:
        print(f"Error loading the configuration file: {e}")
        sys.exit(1)

def process_escape_sequences(color_dict: dict) -> dict:
    """Processes each value in the color dictionary to interpret escape sequences.

    Args:
        color_dict (dict): Dictionary containing color escape sequences.

    Returns:
        dict: Processed color dictionary with interpreted escape sequences.
    """
    for key, value in color_dict.items():
        color_dict[key] = value.encode().decode('unicode_escape')
    return color_dict

def read_data(file_path: str) -> pd.DataFrame:
    """Reads data from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Dataframe containing the CSV data.
    """
    print_output_text_with_color(f"Reading file from {file_path}", 'yellow')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file {file_path} not found.")
    try:
        df = pd.read_csv(file_path)
        print_output_text_with_color(f"Data read successfully. Shape: {df.shape}", 'green')
        return df
    except Exception as e:
        print_output_text_with_color(f"Error reading the file {file_path}: {e}", 'red')
        sys.exit(1)

def feature_engineering(df: pd.DataFrame, cols_to_engineer: dict) -> pd.DataFrame:
    """Performs feature engineering on the dataframe.

    Args:
        df (pd.DataFrame): Input dataframe.
        cols_to_engineer (dict): Dictionary specifying columns to engineer.

    Returns:
        pd.DataFrame: Dataframe with engineered features.
    """
    try:
        print_output_text_with_color("Data Processing Stage 1: Feature engineering", 'yellow')
        for col, (value, new_col) in cols_to_engineer.items():
            df[new_col] = df[col].apply(lambda x: 1 if x == value else 0)
        print_output_text_with_color("Feature engineering completed.", 'green')
        return df
    except Exception as e:
        print_output_text_with_color(f"Error in feature engineering: {e}", 'red')
        sys.exit(1)

def select_columns(df: pd.DataFrame, cols_to_use: list) -> pd.DataFrame:
    """Selects specific columns from the dataframe.

    Args:
        df (pd.DataFrame): Input dataframe.
        cols_to_use (list): List of columns to select.

    Returns:
        pd.DataFrame: Dataframe with selected columns.
    """
    try:
        print_output_text_with_color("Data Processing Stage 2: Selecting specific columns", 'yellow')
        df = df[cols_to_use]
        print_output_text_with_color(f"Columns selected. Shape: {df.shape}", 'green')
        return df
    except Exception as e:
        print_output_text_with_color(f"Error selecting columns: {e}", 'red')
        sys.exit(1)

def one_hot_encode(df: pd.DataFrame, cat_cols: list) -> pd.DataFrame:
    """One-hot encodes categorical columns in the dataframe.

    Args:
        df (pd.DataFrame): Input dataframe.
        cat_cols (list): List of categorical columns to encode.

    Returns:
        pd.DataFrame: Dataframe with one-hot encoded columns.
    """
    try:
        print_output_text_with_color("Data Processing Stage 3: One-hot encoding categorical columns", 'yellow')
        ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
        ohe_df = pd.DataFrame(ohe.fit_transform(df[cat_cols]), columns=ohe.get_feature_names_out(cat_cols), index=df.index)
        df = pd.concat([df, ohe_df], axis=1).drop(columns=cat_cols)
        print_output_text_with_color(f"One-hot encoding completed. Shape: {df.shape}", 'green')
        return df
    except Exception as e:
        print_output_text_with_color(f"Error in one-hot encoding: {e}", 'red')
        sys.exit(1)

def scale_features(df: pd.DataFrame, float_cols: list) -> pd.DataFrame:
    """Scales numerical features using MinMaxScaler.

    Args:
        df (pd.DataFrame): Input dataframe.
        float_cols (list): List of numerical columns to scale.

    Returns:
        pd.DataFrame: Dataframe with scaled numerical features.
    """
    try:
        print_output_text_with_color("Data Processing Stage 4: Scaling numerical features", 'yellow')
        scaler = MinMaxScaler()
        for col in float_cols:
            df[col] = scaler.fit_transform(df[[col]])
        print_output_text_with_color("Feature scaling completed.", 'green')
        return df
    except Exception as e:
        print_output_text_with_color(f"Error scaling features: {e}", 'red')
        sys.exit(1)

def remove_outliers(df: pd.DataFrame, float_cols: list) -> pd.DataFrame:
    """Removes outliers from numerical columns.

    Args:
        df (pd.DataFrame): Input dataframe.
        float_cols (list): List of numerical columns to remove outliers from.

    Returns:
        pd.DataFrame: Dataframe with outliers removed.
    """
    try:
        print_output_text_with_color("Data Processing Stage 5: Removing outliers", 'yellow')
        for col in float_cols:
            lower_limit = df[col].quantile(0.01)
            upper_limit = df[col].quantile(0.99)
            df[col] = df[col].clip(lower=lower_limit, upper=upper_limit)
        print_output_text_with_color("Outliers removed.", 'green')
        return df
    except Exception as e:
        print_output_text_with_color(f"Error removing outliers: {e}", 'red')
        sys.exit(1)

def bin_data(df: pd.DataFrame, col_to_bins: dict) -> pd.DataFrame:
    """Bins data according to the specified binning configuration.

    Args:
        df (pd.DataFrame): Input dataframe.
        col_to_bins (dict): Dictionary specifying columns to bin and bin configuration.

    Returns:
        pd.DataFrame: Dataframe with binned data.
    """
    try:
        print_output_text_with_color("Data Processing Stage 6: Binning data", 'yellow')
        for col, data in col_to_bins.items():
            df[col] = pd.cut(df[col], bins=data['bins'], labels=data['labels'])
        print_output_text_with_color("Binning completed.", 'green')
        return df
    except Exception as e:
        print_output_text_with_color(f"Error in binning data: {e}", 'red')
        sys.exit(1)

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Removes duplicate rows from the dataframe.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Dataframe with duplicates removed.
    """
    try:
        print_output_text_with_color("Data Processing Stage 7: Removing duplicates", 'yellow')
        initial_shape = df.shape
        df = df.drop_duplicates()
        print_output_text_with_color(f"Duplicates removed. {initial_shape[0] - df.shape[0]} rows dropped.", 'green')
        return df
    except Exception as e:
        print_output_text_with_color(f"Error removing duplicates: {e}", 'red')
        sys.exit(1)

def remove_null_values(df: pd.DataFrame) -> pd.DataFrame:
    """Removes rows with null values from the dataframe.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Dataframe with null values removed.
    """
    try:
        print_output_text_with_color("Data Processing Stage 8: Removing null values", 'yellow')
        initial_shape = df.shape
        df = df.dropna()
        print_output_text_with_color(f"Null values removed. {initial_shape[0] - df.shape[0]} rows dropped.", 'green')
        return df
    except Exception as e:
        print_output_text_with_color(f"Error removing null values: {e}", 'red')
        sys.exit(1)

def save_data(df: pd.DataFrame, output_folder: str = "processed_data", test_size: float = 0.2) -> None:
    """Splits the dataframe into training and validation sets, then saves each to a CSV file in separate folders.

    Args:
        df (pd.DataFrame): Dataframe to split and save.
        output_folder (str): Parent folder to save the output CSV files.
        test_size (float): Proportion of the dataset to include in the validation set.
    """
    try:
        # Split the dataframe into training and validation sets
        train_df, val_df = train_test_split(df, test_size=test_size)

        # Create the parent output folder
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Create subfolders for training and validation data
        training_folder = os.path.join(output_folder, "training_data")
        validation_folder = os.path.join(output_folder, "validation_data")

        # if not os.path.exists(training_folder):
        #     os.makedirs(training_folder)
        # if not os.path.exists(validation_folder):
        #     os.makedirs(validation_folder)

        # Generate output file names with current date and time
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        train_file = os.path.join(training_folder, f"train_data_{current_time}.csv")
        val_file = os.path.join(validation_folder, f"val_data_{current_time}.csv")

        # Save the training and validation dataframes to CSV files
        print_output_text_with_color("Data Processing Stage 9: Saving processed training and validation data", 'yellow')
        train_df.to_csv(train_file, index=False)
        val_df.to_csv(val_file, index=False)
        print_output_text_with_color(f"Training data saved to {train_file}.", 'green')
        print_output_text_with_color(f"Validation data saved to {val_file}.", 'green')

    except Exception as e:
        print_output_text_with_color(f"Error saving the files: {e}", 'red')
        sys.exit(1)


def main() -> None:
    """Main function to orchestrate data processing."""
    global current_file_name
    global color_codes
    
    # Load configuration
    config = load_config()

    # Get parameters from the configuration file
    input_file = config['PARAMETERS']['INPUT_FILE']
    cols_to_use = config['PARAMETERS']['COLS_TO_USE']
    float_cols = config['PARAMETERS']['FLOAT_COLS']
    cat_cols = config['PARAMETERS']['CAT_COLS']
    cols_to_bin = config['PARAMETERS']['COLS_TO_BIN']
    current_file_name = config['PARAMETERS']['FILE_NAME_DATAPROCESSING']
    color_codes = config['PARAMETERS']['COLOR_DICT']
    cols_to_engineer = config['PARAMETERS']['COLS_TO_FEATURE_ENGINEER']
    output_folder = config['PARAMETERS']['SAVED_PROCESSED_PATH']
    test_size = float(config['PARAMETERS']['TEST_SIZE'])
    
    # Process the escape sequences
    color_codes = process_escape_sequences(color_codes)
    
    # Read data
    data = read_data(input_file)

    # Perform feature engineering
    data = feature_engineering(data, cols_to_engineer)

    # Select necessary columns
    data = select_columns(data, cols_to_use)

    # Removing duplicates
    data = remove_duplicates(data)
    
    # Removing null values
    data = remove_null_values(data)

    # Removing outliers
    data = remove_outliers(data, float_cols)

    # Bin the values
    data = bin_data(data, cols_to_bin)

    # One-hot encode categorical columns
    data = one_hot_encode(data, cat_cols)

    # Scale numerical features
    data = scale_features(data, float_cols)

    # Save the processed data
    save_data(data, output_folder, test_size)

if __name__ == "__main__":
    while True:
        current_stage = 0  # Reset num to 0 at the start of each new round
        print("\nStarting a new round of execution...")
        main()
        print("Finished execution.")
        print("Waiting 1 minute before the next round...")
        time.sleep(60)





