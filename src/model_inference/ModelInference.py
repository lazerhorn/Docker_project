import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
import yaml

# Global variables
current_stage = 1
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

    print(f'{color_codes["blue"]}({current_file_name}): {color_codes[color]}{str(current_stage) + "/4"} {text} {color_codes["reset"]}')
    
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

def load_latest_validation_data(folder_path: str, target_col: str):
    """Loads the latest CSV file containing validation data from a specified folder.

    Args:
        folder_path (str): Path to the folder containing the CSV files.
        target_col (str): The name of the target column.

    Returns:
        tuple: Features (X_val) and target (y_val) from the latest validation data file.
    """
    try:
        # Find the latest file in the folder
        files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        latest_file = max([os.path.join(folder_path, f) for f in files], key=os.path.getmtime)
        
        print_output_text_with_color(f"Loading validation data from the latest file: {latest_file}", 'yellow')
        data = pd.read_csv(latest_file)
        X_val = data.drop(target_col, axis=1)
        y_val = data[target_col]
        
        print_output_text_with_color("Validation data successfully loaded.", 'green')
        return X_val, y_val
    except Exception as e:
        print_output_text_with_color(f"Error processing the validation data from folder {folder_path}: {e}", 'red')
        sys.exit(1)


def evaluate_model(model_name: str, model, X: pd.DataFrame, y: pd.Series, dataset_name: str) -> None:
    """Evaluates the model on the specified dataset.

    Args:
        model_name (str): The name of the model.
        model: Trained model.
        X (pd.DataFrame): Features of the dataset.
        y (pd.Series): True labels of the dataset.
        dataset_name (str): Name of the dataset ('Test' or 'Validation').
    """
    try:
        print_output_text_with_color(f"Evaluating model {model_name} on {dataset_name} set", 'yellow')
        y_pred = model.predict(X)
        print_output_text_with_color(f"Evaluation results for {dataset_name} set:", 'green')
        get_model_accuracy(model_name, y, y_pred)
        get_model_classification_report(model_name, y, y_pred)
        get_confusion_matrix(model_name, y, y_pred, plot_heatmap=True)
        check_thresholds(model_name, y, y_pred)
    except Exception as e:
        print_output_text_with_color(f"Error evaluating the model {model_name} on {dataset_name} set: {e}", 'red')
        sys.exit(1)


def get_model_accuracy(model_name: str, y_true: pd.Series, y_pred: np.ndarray) -> None:
    """Calculates and prints the accuracy of the model.

    Args:
        model_name (str): The name of the model.
        y_true (pd.Series): True labels.
        y_pred (np.ndarray): Predicted labels.
    """
    accuracy = accuracy_score(y_true, y_pred)
    print_output_text_with_color(f"(model {model_name}): Accuracy", 'green')
    print(accuracy)

def get_model_classification_report(model_name: str, y_true: pd.Series, y_pred: np.ndarray) -> dict:
    """Generates and prints the classification report.

    Args:
        model_name (str): The name of the model.
        y_true (pd.Series): True labels.
        y_pred (np.ndarray): Predicted labels.

    Returns:
        dict: Classification report.
    """
    report = classification_report(y_true, y_pred, output_dict=True)
    print_output_text_with_color(f"(model {model_name}): Classification Report", 'green')
    print("Detailed metrics:")
    print(report)
    print(f"Precision: {report['weighted avg']['precision']:.4f}")
    print(f"Recall: {report['weighted avg']['recall']:.4f}")
    print(f"F1-Score: {report['weighted avg']['f1-score']:.4f}")
    return report

def get_confusion_matrix(model_name: str, y_true: pd.Series, y_pred: np.ndarray, plot_heatmap: bool = False) -> None:
    """Generates and optionally plots the confusion matrix.

    Args:
        model_name (str): The name of the model.
        y_true (pd.Series): True labels.
        y_pred (np.ndarray): Predicted labels.
        plot_heatmap (bool): Whether to plot the confusion matrix as a heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    print_output_text_with_color(f"(model {model_name}): Confusion matrix", 'green')
    print(cm)
    if plot_heatmap:
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()

def check_thresholds(model_name: str, y_true: pd.Series, y_pred: np.ndarray) -> None:
    """Checks if the model meets the required performance thresholds.

    Args:
        model_name (str): The name of the model.
        y_true (pd.Series): True labels.
        y_pred (np.ndarray): Predicted labels.
    """
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    weighted_avg = report['weighted avg']
    precision = weighted_avg['precision']
    recall = weighted_avg['recall']
    f1_score = weighted_avg['f1-score']
    
    if accuracy >= 0.60:
        print_output_text_with_color(f"(model {model_name}): Accuracy meets the threshold (60%).", 'green')
    else:
        print_output_text_with_color(f"(model {model_name}): Accuracy does not meet the threshold (60%).", 'red')
    
    if precision >= 0.60:
        print_output_text_with_color(f"(model {model_name}): Precision meets the threshold (60%).", 'green')
    else:
        print_output_text_with_color(f"(model {model_name}): Precision does not meet the threshold (60%).", 'red')
    
    if recall >= 0.60:
        print_output_text_with_color(f"(model {model_name}): Recall meets the threshold (60%).", 'green')
    else:
        print_output_text_with_color(f"(model {model_name}): Recall does not meet the threshold (60%).", 'red')
    
    if f1_score >= 0.70:
        print_output_text_with_color(f"(model {model_name}): F1-score meets the threshold (70%). The most important metric.", 'green')
    else:
        print_output_text_with_color(f"(model {model_name}): F1-score does not meet the threshold (70%).", 'red')

    if accuracy >= 0.60 and precision >= 0.60 and recall >= 0.60 and f1_score >= 0.70:
        print_output_text_with_color(f"(model {model_name}): Model meets all the required thresholds.", 'green')
    else:
        print_output_text_with_color(f"(model {model_name}): Model does not meet all the required thresholds.", 'red')
        print(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-score: {f1_score}')

def load_latest_model(save_dir: str, model_name: str):
    """Loads the latest saved model from the specified subfolder within the directory.

    Args:
        save_dir (str): Parent directory containing the model subfolders.
        model_name (str): The name of the model ('rf' for Random Forest, 'xgb' for XGBoost).

    Returns:
        model: The latest loaded model.
    """
    try:
        # Determine the subfolder based on the model name
        if model_name.lower() == "rf":
            subfolder = "model_rf"
        elif model_name.lower() == "xgb":
            subfolder = "model_xgb"
        else:
            raise ValueError("Model name not recognized. Please use 'rf' for Random Forest or 'xgb' for XGBoost.")

        # Create the full path to the subfolder
        full_save_dir = os.path.join(save_dir, subfolder)
        
        # Find the latest model file in the subfolder
        files = [f for f in os.listdir(full_save_dir) if f.endswith('.pkl')]
        latest_file = max([os.path.join(full_save_dir, f) for f in files], key=os.path.getmtime)
        
        # Load and return the latest model
        with open(latest_file, 'rb') as f:
            model = pickle.load(f)
        
        print_output_text_with_color(f"Loaded the latest model from {latest_file}", 'green')
        return model
    
    except ValueError as ve:
        print_output_text_with_color(str(ve), 'red')
        sys.exit(1)
    except Exception as e:
        print_output_text_with_color(f"Error loading the latest model: {e}", 'red')
        sys.exit(1)


def main():
    global current_file_name
    global color_codes

    # Load configuration parameters
    config = load_config()

    processed_data_path_test = config['PARAMETERS']['PROCESSED_FILE_VALIDATE']
    saved_models_path = config['PARAMETERS']['SAVED_MODELS_PATH']
    models_to_train = config['PARAMETERS']['MODELS_TO_TRAIN']
    current_file_name = config['PARAMETERS']['FILE_NAME_MODELINFERENCE']
    color_codes = config['PARAMETERS']['COLOR_DICT']
    target_col = config['PARAMETERS']['TARGET_COL']
    color_codes = process_escape_sequences(color_codes)


    # Get training, testing, and validation datasets
    X_val, y_val = load_latest_validation_data(processed_data_path_test, target_col)

    for model_name in models_to_train:
        model = load_latest_model(saved_models_path, model_name)
        evaluate_model(model_name, model, X_val, y_val, 'Validation')
        print_output_text_with_color(f"(model {model_name}): Model evaluation done!", 'green')

if __name__ == "__main__":
    main()
