import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import pickle
import yaml
from datetime import datetime
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

    print(f'{color_codes["blue"]}({current_file_name}): {color_codes[color]}{str(current_stage) + "/5"} {text} {color_codes["reset"]}')
    
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

def load_latest_training_data(folder_path: str, target_col: str):
    """Loads the latest CSV file containing training data from a specified folder.

    Args:
        folder_path (str): Path to the folder containing the CSV files.
        target_col (str): The name of the target column.

    Returns:
        tuple: Features (X_train) and target (y_train) from the latest training data file.
    """
    try:
        # Find the latest file in the folder
        files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        latest_file = max([os.path.join(folder_path, f) for f in files], key=os.path.getmtime)
        
        print_output_text_with_color(f"Loading training data from the latest file: {latest_file}", 'yellow')
        data = pd.read_csv(latest_file)
        X_train = data.drop(target_col, axis=1)
        y_train = data[target_col]
        
        print_output_text_with_color("Training data successfully loaded.", 'green')
        return X_train, y_train
    except Exception as e:
        print_output_text_with_color(f"Error processing the training data from folder {folder_path}: {e}", 'red')
        sys.exit(1)

def train_model(model_name: str, X_train: pd.DataFrame, y_train: pd.Series, model_hyperparameters: dict):
    """Trains a specified model with the given hyperparameters.

    Args:
        model_name (str): The name of the model ('rf' or 'xgb').
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        model_hyperparameters (dict): Hyperparameters for the model.

    Returns:
        model: Trained model.
    """
    try:
        print_output_text_with_color(f"Training model {model_name}", 'yellow')
        if model_name == 'rf':
            model = RandomForestClassifier(**model_hyperparameters)
        elif model_name == 'xgb':
            model = xgb.XGBClassifier(**model_hyperparameters)
        model.fit(X_train, y_train)
        print_output_text_with_color(f"Training model {model_name} successfully", 'green')
        return model
    except Exception as e:
        print_output_text_with_color(f"Error training the model {model_name}: {e}", 'red')
        sys.exit(1)


def save_model(model, model_name: str, save_dir: str) -> None:
    """Saves the trained model to a specified subfolder within the directory.

    Args:
        model: Trained model.
        model_name (str): The name of the model.
        save_dir (str): Parent directory to save the model.

    Raises:
        ValueError: If the model_name is not recognized.
    """
    try:
        print_output_text_with_color(f"Saving model {model_name}", 'yellow')
        
        # Determine the subfolder based on the model name
        if model_name.lower() == "rf":
            subfolder = "model_rf"
        elif model_name.lower() == "xgb":
            subfolder = "model_xgb"
        else:
            raise ValueError("Model name not recognized. Please use 'rf' for Random Forest or 'xgb' for XGBoost.")

        # Create the full save directory path
        full_save_dir = os.path.join(save_dir, subfolder)
        if not os.path.exists(full_save_dir):
            os.makedirs(full_save_dir)
        
        # Save the model with a timestamp
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(full_save_dir, f'{current_time}_{model_name}.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(model, f)
        
        print_output_text_with_color(f"Model {model_name} saved to {save_path}", 'green')
    
    except ValueError as ve:
        print_output_text_with_color(str(ve), 'red')
        sys.exit(1)
    except Exception as e:
        print_output_text_with_color(f"Error saving the model {model_name}: {e}", 'red')
        sys.exit(1)

def main():
    global current_file_name
    global color_codes

    # Load configuration parameters
    config = load_config()

    processed_data_path_train = config['PARAMETERS']['PROCESSED_FILE_TRAIN']
    saved_models_path = config['PARAMETERS']['SAVED_MODELS_PATH']
    models_to_train = config['PARAMETERS']['MODELS_TO_TRAIN']
    model_hyperparameters = config['PARAMETERS']['MODEL_HYPERPARAMETERS']
    current_file_name = config['PARAMETERS']['FILE_NAME_DATAMODELING']
    color_codes = config['PARAMETERS']['COLOR_DICT']
    target_col = config['PARAMETERS']['TARGET_COL']
    color_codes = process_escape_sequences(color_codes)


    # Get training, testing, and validation datasets
    X_train, y_train = load_latest_training_data(processed_data_path_train, target_col)

    for model_name in models_to_train:
        model = train_model(model_name, X_train, y_train, model_hyperparameters[model_name])
        # evaluate_model(model_name, model, X_train, y_train, 'training')
        # evaluate_model(model_name, model, X_val, y_val, 'Validation')
        save_model(model, model_name, saved_models_path)
        # print_output_text_with_color(f"(model {model_name}): Model evaluation done!", 'green')

if __name__ == "__main__":
    while True:
        current_stage = 0  # Reset num to 0 at the start of each new round
        print("\nStarting a new round of execution...")
        main()
        print("Finished execution.")
        print("Waiting 1 minute before the next round...")
        time.sleep(60)

