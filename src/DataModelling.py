import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split
import yaml
from datetime import datetime

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

    print(f'{color_codes["blue"]}({current_file_name}): {color_codes[color]}{str(current_stage) + "/10"} {text} {color_codes["reset"]}')
    
def load_config(config_file: str = 'config.yml') -> dict:
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

def get_train_test_val_datasets(file_path: str, test_size: float, validation_size: float, target_col: str):
    """Reads data from a CSV file and splits it into training, testing, and validation sets.

    Args:
        file_path (str): Path to the CSV file.
        test_size (float): Proportion of the dataset to include in the test split.
        validation_size (float): Proportion of the dataset to include in the validation split.
        target_col (str): The name of the target column.

    Returns:
        tuple: Split datasets (X_train, X_test, X_val, y_train, y_test, y_val).
    """
    try:
        print_output_text_with_color(f"Reading file from {file_path}", 'yellow')
        data = pd.read_csv(file_path)
        X = data.drop(target_col, axis=1)
        y = data[target_col]
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(test_size + validation_size), random_state=42)
        X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=(validation_size / (test_size + validation_size)), random_state=42)
        print_output_text_with_color(f"Data split into train, test, and validation sets with sizes: {X_train.shape}, {X_test.shape}, {X_val.shape}", 'green')
        return X_train, X_test, X_val, y_train, y_test, y_val
    except Exception as e:
        print_output_text_with_color(f"Error reading the file {file_path}: {e}", 'red')
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

def save_model(model, model_name: str, save_dir: str) -> None:
    """Saves the trained model to the specified directory.

    Args:
        model: Trained model.
        model_name (str): The name of the model.
        save_dir (str): Directory to save the model.
    """
    try:
        print_output_text_with_color(f"Saving model {model_name}", 'yellow')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_dir, f'{current_time}_{model_name}.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(model, f)
        print_output_text_with_color(f"Model {model_name} saved to {save_path}", 'green')
    except Exception as e:
        print_output_text_with_color(f"Error saving the model {model_name}: {e}", 'red')
        sys.exit(1)

def main():
    global current_file_name
    global color_codes

    # Load configuration parameters
    config = load_config()

    processed_data_path = config['PARAMETERS']['PROCESSED_FILE']
    saved_models_path = config['PARAMETERS']['SAVED_MODELS_PATH']
    test_size = float(config['PARAMETERS']['TEST_SIZE'])
    validation_size = float(config['PARAMETERS']['VALIDATION_SIZE'])
    models_to_train = config['PARAMETERS']['MODELS_TO_TRAIN']
    model_hyperparameters = config['PARAMETERS']['MODEL_HYPERPARAMETERS']
    current_file_name = config['PARAMETERS']['FILE_NAME_DATAMODELING']
    color_codes = config['PARAMETERS']['COLOR_DICT']
    target_col = config['PARAMETERS']['TARGET_COL']
    color_codes = process_escape_sequences(color_codes)

    # Get training, testing, and validation datasets
    X_train, X_test, X_val, y_train, y_test, y_val = get_train_test_val_datasets(processed_data_path, test_size, validation_size, target_col)

    for model_name in models_to_train:
        model = train_model(model_name, X_train, y_train, model_hyperparameters[model_name])
        evaluate_model(model_name, model, X_test, y_test, 'Test')
        evaluate_model(model_name, model, X_val, y_val, 'Validation')
        save_model(model, model_name, saved_models_path)
        print_output_text_with_color(f"(model {model_name}): Model evaluation done!", 'green')

if __name__ == "__main__":
    main()
