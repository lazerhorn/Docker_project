---
Full Name: Tan Jun Wei
Email address: 221628L@mymail.nyp.edu.sg

---
# PetFinder Project for Associate AI Engineer Technical Test 

### Background
PetFinder.my is one of the leading pet adoption portals in Malaysia and a non-profit organisation for animal welfare. However, in recent years, it has faced falling donations and slower adoption rates. The organisation is working on a new business model that will allow it to be more self-sutstainable and is looking at ways to increase revenue through advertising and sponsorship. The key metric for the organisation is the adoption rate, and with higher adoption rate, the organization can have continuous fresh new contents for the portal, and which in turn help to boost its revenue from sponsorship and partnership. More importantly, better adoption rate means more animals can find new home sooner. 

### Objective
To predict adoption rate and better understand the adopter’s preferences.

### Adoption Speed
Based on the description of the dataset, there are 5 types of pet adoption speed, they are categorised as such:

AdoptionSpeed

- 0 - Pet was adopted on the same day as it was listed.
- 1 - Pet was adopted between 1 and 7 days (1st week) after being listed.
- 2 - Pet was adopted between 8 and 30 days (1st month) after being listed.
- 3 - Pet was adopted between 31 and 90 days (2nd & 3rd month) after being listed.
- 4 - No adoption after 100 days of being listed. (There are no pets in this dataset that waited between 90 and 100 days).

This will be our target variable for the Machine Learning Algorithm.

### Project Deliverables

1. **Exploratory Data Analysis (EDA) Report**:
   - Conduct EDA to gain insights into the factors influencing pet adoption and the speed of adoption.
   - Identify key features impacting whether a pet gets adopted and the rate at which it happens.

2. **Complete Machine Learning Pipeline**:
   - **Data Preparation**: 
     - Clean and preprocess the dataset based on insights from EDA.
     - Engineer features to enhance the dataset's predictive power.
   - **Data Modelling**:
     - Split the processed dataset into training, testing, and validation sets.
     - Train two machine learning models, RandomForestClassifier and XGBoostClassifier, using these datasets.
     - Evaluate model performance using standard ML metrics such as accuracy, precision, recall, and F1-score.
     - Compare the performance of both models to determine the more accurate predictor of pet adoption speed.

---
## Summary of Key Findings (EDA.ipnb/ EDA.pdf)
An EDA was conducted and the detailed analysis can be found under eda.ipynb/ eda.pdf.
Here is a summary of the key insights that was gained while doing the EDA. These insights have influenced the feature selection and feature engineering process.

This EDA notebook consist 5 main sections :
1. Basic Data understanding
2. Basic data cleaning on Main Dataset (pets_prepared.csv)
3. Main DataSet analysis
4. Analysis of Column Features and Their Impact on Adoption Rate and Speed
5. Summary of the different features that will affect the adoption speed and rate

### 1. Basic Data Understanding

- **Dataset Overview**:
  - The `pets_prepared.csv` dataset consists of 14,993 pet listings with 48 features (excluding the `AdoptionSpeed` column).
  - Only two types of animals are included: cats and dogs.
  - Features with missing values: **`Name`**, **`Description`**, and **`BreedName`**.

- **Supplementary Datasets**:
  - `color_labels.csv`: Contains the ID for color-related columns.
  - `state_labels.csv`: Contains the ID for state-related columns.
  - `breed_labels.csv`: Contains the ID for breed-related columns.

### 2. Basic Data Cleaning on Main Dataset (pets_prepared.csv)

- **General Data Cleaning**:
  - Number of rows dropped: 25
  - Percentage of rows dropped: 0.17%
  - No duplicated rows in the dataframe.

- **Outlier Removal**:
  - Total rows removed after removing outliers: 35
  - Percentage of rows removed after removing outliers: 0.23%

- **Erroneous Value Removal**:
  - Total rows removed after removing erroneous values: 0
  - Percentage of rows removed after removing erroneous values: 0.00% (no erroneous values)

- **Inconsistency in Column Values**:
  - **AgeBins** and **FeeBins** exhibit inconsistencies in their formatting and range representation:
    - AgeBins includes ranges enclosed in square brackets [ ] and parentheses ( ).
    - FeeBins has mixed representations with some ranges missing brackets entirely.
    - These inconsistencies could potentially affect data analysis and visualizations.

### 3. Main Dataset Analysis

**Dataset Composition:**
- **Animals:** 54.2% dogs, 45.8% cats.
- **Adoption Speed:**
  - Types 1, 2, 3, and 4: ~97%
  - Type 0: 2.73%

**Implications for ML Model Training:**
- **Imbalance:** Low percentage for adoption speed type 0 may affect model accuracy for this category.

**Feature Analysis on continuous data:**
- **Age:**
  - Most pets: 0-25 months
  - Outliers: >25 to >175 months
- **Quantity:**
  - Mostly 0 or 1
  - Outliers: 2 to 16
- **Fee:**
  - Mostly 0
  - Outliers: >0 to 800
- **VideoAmt:**
  - Mostly 0
  - Outliers: 1 to 5
- **PhotoAmt:**
  - Mostly <10
  - Outliers: 10 to 30

**Implications:**
- **Feature Variability:** `Quantity`, `Fee`, and `VideoAmt` show little variability; may not be useful for prediction.
- **Outliers:** Present in continuous features; need addressing in data pipeline.

### Correlation Heatmap Insights

**Low Correlation with Adoption Speed and Status:**
- `VideoAmt`, `State`, `Fee`, `Dewormed`, `Color3`: Correlations 0.00 to ±0.01.

**Relatively Higher Correlation with Adoption Speed and Status:**
- `BreedPure`, `Sterilized`, `FurLength`, `Breed1`, `Age`, `Type`: Correlations ±0.08 to ±0.11.

**PhotoAmt Correlation:**
- Low with adoption speed; higher with adopted status (0.10).

**Redundant Features:**
- High correlations between `Dewormed`, `Vaccinated`, and `Sterilized`.

### Insights:
- **Significant Features:** `Breed`, `Age`, `Type`, `FurLength`, `Sterilized` are key for predicting adoption speed.
- **Redundancies:** Identifying redundant features helps streamline the analysis.

### 4/5. Analysis of Column Features and Their Impact on Adoption Rate and Speed / Summary

### **Part 5:** Summary

**Key Takeaways:**
- **Younger pets** are more likely to be adopted and adopted faster than older pets.
- **Cats** are slightly more likely to be adopted and adopted faster compared to dogs.
- **Male pets** are more likely to be adopted and adopted faster than female and mixed-gender pets.
- **Purebred animals** are more likely to be adopted and adopted faster than mixed breeds.
- **Pets with more photos and videos** have higher adoption rates.
- **Healthier pets** are substantially more likely to be adopted and adopted faster.
- **Pets that are not sterilized** are more likely to be adopted and adopted faster than sterilized or uncertain pets.
- **Pets with long hair** are more likely to be adopted and adopted faster than those with medium or short hair.
- **Named pets** are more likely to be adopted and adopted faster than unnamed pets.
- **Price does not** significantly affect adoption likelihood; high-priced and low-priced pets have similar adoption rates.

**Feature Selection:**
The following features will be dropped due to low impact on adoption speed or difficulty in use for training:
- Fee (little variance and low correlation)
- VideoAmt (little variance and low correlation)
- Name (text data is hard to use; NameorNO will be used instead)
- Description (text data is hard to use)

---
## How to execute the pipeline and modify the configurations / parameters
To execute the pipeline, there are 5 steps: 
First, navigate to the root directory of this project (CAIEsubmission) .
Then,
1. Create a virtual environment using python 3.7 
    - `py -3.7 -m venv venv` (Git Bash)
2. Activate the virtual environment to create a isolated environment:
    - `source venv/scripts/activate` (Git Bash)
3. Install the required libraries to run the pipeline. 
    - `pip install -r requirements.txt` (Git Bash)
4. Import the dataset into the raw_data folder
    - copy the `pets_prepared.csv` into the raw_data folder
5. Run the `run.sh` file
    - check if the file has executable permissions by running the command in your terminal: `ls -l run.sh`
    - if the file has no exectuable permission, run this instead: `chmod +x run.sh`
    - finally, run this command: `bash run.sh`

By running the above commands, addtional folders such as `processed_data` and `saved_model` will be created and data preparation and model training pipeline will be executed.
To modify the configurations needed for the data preparation and machine learning pipeline, the parameters listed under `config.yml` can be modified. 

For example, 
if there is a need to change the hyperparameter of the RandomForestClassifier, navigate to the `config.yml` file and change the hyperparameters listed accordingly.

---
## Flow of Pipeline

This project's pipeline comprises two main components:

### 1. Data Preparation Pipeline (DataProcessing.py)

1. **Loading and Filtering Data:**
   - The pipeline begins by loading the raw dataset `pets_prepared.csv` from the `raw_data` folder.
   - The dataset is filtered based on the specified features essential for training machine learning models, as defined in `config.yml`.

2. **Feature Engineering:**
   - Relevant feature engineering techniques are applied to enhance the dataset. For instance, categorical variables are transformed using one-hot encoding, and specific binary features are engineered as per configuration.

3. **Data Transformation and Cleaning:**
   - Selected columns undergo transformation and cleaning stages to ensure data integrity and model readiness.
   - Duplicates and null values are removed to streamline the dataset for further analysis.

4. **Handling Numerical Features:**
   - Numerical features are scaled using MinMaxScaler to normalize their values and improve model performance.
   - Outliers in numerical columns are identified and clipped to maintain data consistency and model robustness.

5. **Binning Data:**
   - Certain numerical columns are binned according to predefined configurations in `config.yml`, facilitating easier analysis and model interpretation.

6. **Saving Processed Data:**
   - Upon completion of data preparation steps, the processed dataset is saved in the `processed_data` folder.
   - Each saved file is uniquely labeled with a timestamp to maintain version control and facilitate traceability.


### 2. Machine Learning Model Training and Evaluation Pipeline (DataModelling.py)

1. **Model Training and Evaluation:**
   - In this pipeline, the focus is on training machine learning models specified in `models_to_train` from the configuration file (`config.yml`).
   - The models are trained using datasets split from the output of the Data Preparation Pipeline (processed dataset).

2. **Dataset Splitting:**
   - The processed dataset is divided into three distinct sets:
     - **Training Set:** Used to train the machine learning models (`X_train` and `y_train`).
     - **Validation Set:** Employed to fine-tune model hyperparameters and assess generalizability (`X_val` and `y_val`).
     - **Testing Set:** Utilized to evaluate the final model performance on unseen data (`X_test` and `y_test`).

3. **Training Process:**
   - Each model undergoes training on the training dataset (`X_train`) using hyperparameters defined in `config.yml`.
   - After training, the trained models are saved in the `saved_models` directory for future use and reference.

4. **Evaluation Metrics:**
   - Post-training, the models' performance is evaluated using metrics such as accuracy, precision, recall, and F1-score.
   - A detailed classification report is generated, containing these metrics, providing insights into the model's predictive capabilities.
   - Additionally, a confusion matrix is produced to visualize the model's performance in terms of true positive, true negative, false positive, and false negative predictions.

5. **Threshold Check:**
   - The pipeline includes a check against predefined performance thresholds (e.g., accuracy ≥ 60%, precision ≥ 60%, recall ≥ 60%, and F1-score ≥ 70%).
   - Models that meet these thresholds are considered acceptable for deployment or further optimization.

6. **Output and Logging:**
   - Throughout the pipeline, colored output messages indicate the progress of each stage, ensuring transparency and facilitating debugging.

---
### Explanation of Choice of Models

1. **Nature of the Dataset and Task**:
   - The dataset provided for this project includes features that are primarily categorical or binary, with a target variable (`AdoptionSpeed`) that has five distinct classes (0, 1, 2, 3, 4). This sets up a supervised multi-class classification task where the goal is to predict the adoption speed of pets.
   
2. **Classification Requirement**:
   - Given the task's nature as a multi-class classification problem, where we need to classify instances into one of five categories, classifiers are a suitable choice. Classifiers are designed specifically to assign labels to instances based on their features, making them well-suited for this prediction task.

3. **Non-Linearity in Data**:
   - Many of the features in the dataset are categorical or binary, such as `BreedName`, which does not exhibit a linear relationship with `AdoptionSpeed`. Non-linear classifiers are advantageous in scenarios where the relationship between features and the target variable is complex and cannot be adequately captured by linear models. Both RandomForestClassifier and XGBoostClassifier are capable of capturing non-linear relationships between features and the target variable, making them appropriate choices for this dataset.

4. **RandomForestClassifier**:
   - **Advantages**: RandomForestClassifier is an ensemble learning method that operates by constructing multiple decision trees during training and outputs the mode of the classes (for classification tasks) or mean prediction (for regression tasks) of individual trees. It excels in handling high-dimensional datasets with many categorical features and can capture complex interactions and non-linearities effectively.
   - **Applicability**: Due to its robustness to noise and overfitting, RandomForestClassifier is often chosen for tasks where interpretability of individual trees and robustness against outliers are desired. It can handle missing data and maintain accuracy even when a large proportion of the data is missing.

5. **XGBoostClassifier**:
   - **Advantages**: XGBoost (Extreme Gradient Boosting) is an optimized distributed gradient boosting library designed to be highly efficient, flexible, and portable. It builds trees sequentially, where each tree corrects the errors of its predecessor, resulting in boosted models that can achieve higher accuracy than individual trees alone.
   - **Applicability**: XGBoost is known for its performance on structured/tabular data, making it suitable for datasets like `pets_prepared` where features are well-defined and categorical. It handles missing data internally and is robust to outliers, often outperforming other models due to its regularization techniques and ability to capture complex interactions.

6. **Performance and Documentation**:
   - Both RandomForestClassifier and XGBoostClassifier have been extensively documented and proven effective across a wide range of applications and datasets. Their performance on tabular data, particularly in classification tasks with categorical features, is well-documented in literature and practical applications.

7. **Choice Rationale**:
   - Given the dataset characteristics, the need for handling categorical features, and the documented performance advantages, RandomForestClassifier and XGBoostClassifier were selected as they are expected to provide robust predictions for the `AdoptionSpeed` classification task. These models not only address the non-linearity in the data but also leverage ensemble methods to improve predictive accuracy and generalize well to unseen data.

---
## Evaluation of models developed


### Evaluation of Models Developed

The performance of the models developed is evaluated based on four metrics:

1. **F1 Score**:
   - The F1 score is a measure of the model's accuracy that takes into account both precision and recall. It is the harmonic mean of precision and recall and is especially useful when classes are imbalanced, as it gives equal weight to both precision and recall.

2. **Recall**:
   - Recall is the proportion of positive instances that are correctly identified by the model. In multi-class classification, recall measures the proportion of true positives for each class. It is calculated as the ratio of true positives for a class to the sum of true positives and false negatives for that class.

3. **Precision**:
   - Precision is the proportion of positive predictions that are correct. In multi-class classification, precision measures the proportion of true positives for each class out of all instances predicted to be positive for that class. It is calculated as the ratio of true positives for a class to the sum of true positives and false positives for that class.

4. **Accuracy**:
   - Accuracy is the proportion of correct predictions made by the model. In multi-class classification, accuracy measures the proportion of correct predictions across all classes. It is calculated as the ratio of the number of correct predictions to the total number of predictions made.

#### Classification Report of RF (Random Forest):

```
              precision    recall  f1-score   support

Class 0       1.000      0.008     0.016       124
Class 1       0.344      0.302     0.321       892
Class 2       0.313      0.384     0.345      1187
Class 3       0.362      0.160     0.222       949
Class 4       0.459      0.642     0.535      1237

Accuracy                           0.381      4389
Macro average      0.496      0.299     0.288      4389
Weighted average   0.390      0.381     0.358      4389
```

#### Classification Report of XGB (XGBoost):

```
              precision    recall  f1-score   support

Class 0       1.00      0.01      0.02       124
Class 1       0.34      0.30      0.32       892
Class 2       0.31      0.38      0.35      1187
Class 3       0.36      0.16      0.22       949
Class 4       0.46      0.64      0.54      1237

Accuracy                          0.38      4389
Macro average     0.50      0.30      0.29      4389
Weighted average  0.39      0.38      0.36      4389
```

### Model Comparison

Comparing the two models, it can be observed that the Random Forest (RF) model performs slightly better than the XGBoost (XGB) model in terms of precision, recall, and F1-score for each class, as well as the weighted averages of these metrics. However, the difference in performance between the two models is relatively small.

For example, the recall for Class 4 is 0.642 for RF and 0.64 for XGB, indicating that RF performs slightly better on this class. However, in terms of overall accuracy, RF slightly outperforms XGB with 38.1% accuracy compared to XGB's 38.0% accuracy.

### Recommendations for Improvement

Further hyperparameter tuning can be carried out by modifying the parameters listed under `config/constants.py` to fine-tune the model further. This may involve adjusting parameters such as learning rate, maximum depth, number of estimators, and other model-specific parameters to optimize performance.



---
### Folder structure:
    CAIEsubmission/
    │
    ├── .ipynb_checkpoints/
    │
    ├── proccessed_data/
    │
    ├── raw_data/
    │
    ├── saved_model/
    │
    ├── src/
    │   ├── DataModelling.py
    │   ├── DataProcessing.py
    │
    ├── venv/
    │
    ├── config.yml
    ├── eda.ipynb
    ├── eda.pdf
    ├── README copy.md
    ├── README.md
    ├── requirements.txt
    └── run.sh


---
### Programming Lnaguage used: Python 3.7.1
List of libraries used:
    pandas==1.3.5
    seaborn==0.12.2
    matplotlib==3.5.3
    plotly==5.18.0
    scikit-learn==1.0.2
    xgboost==1.1.0
    pyyaml==6.0.1
    numpy==1.21.6
    scipy==1.7.3
    ipykernel==6.16.2
    wordcloud==1.9.3