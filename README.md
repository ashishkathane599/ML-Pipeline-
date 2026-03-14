# ⚙️ End-to-End Machine Learning Pipeline with Scikit-Learn

This project demonstrates how to build a **complete machine learning pipeline using Scikit-Learn** that automates the entire ML workflow — from preprocessing to model training and hyperparameter tuning.

Instead of performing preprocessing and modeling steps separately, everything is integrated into a **single reproducible pipeline**.

The goal of this repository is to showcase **production-style ML workflow design** using Scikit-Learn Pipelines.

---

# 📌 Why Machine Learning Pipelines?

In real-world ML projects, data must go through multiple steps before a model can be trained:

- Handling missing values  
- Encoding categorical variables  
- Feature scaling  
- Feature selection  
- Model training  

If these steps are implemented separately, it often leads to:

❌ Data leakage  
❌ Inconsistent preprocessing between training and prediction  
❌ Hard-to-maintain code  
❌ Repeated manual steps  
❌ Deployment issues  

A **Machine Learning Pipeline solves these problems** by chaining all steps together.

---

# 🧠 Pipeline Workflow

The pipeline in this project follows the following sequence:

```
Missing Value Handling
        ↓
Categorical Encoding
        ↓
Feature Scaling
        ↓
Feature Selection
        ↓
Model Training
```

Each step is executed **automatically in sequence**.

---

# ⚙️ Pipeline Components

## 1️⃣ Missing Value Handling

Missing values are handled using:

```
SimpleImputer
```

- Numerical features → Mean Imputation  
- Categorical features → Most Frequent Imputation  

---

## 2️⃣ Categorical Encoding

Categorical variables are transformed using:

```
OneHotEncoder
```

Benefits:

- Converts categorical data into numerical format
- Prevents ordinal bias
- Works well with most ML algorithms

---

## 3️⃣ Feature Scaling

Feature scaling is applied using:

```
MinMaxScaler
```

This scales values between **0 and 1**, improving model performance and stability.

---

## 4️⃣ Feature Selection

Feature selection is performed using:

```
SelectKBest
Score Function : Chi-Square
```

Benefits:

- Removes irrelevant features
- Reduces model complexity
- Improves generalization

---

## 5️⃣ Model Training

The final step in the pipeline trains the model:

```
DecisionTreeClassifier
```

---

# 🏗️ Complete Pipeline Implementation

```python
pip = Pipeline([
    ('trf1', trf1),   # Missing value handling
    ('trf2', trf2),   # Encoding
    ('trf3', trf3),   # Scaling
    ('trf4', trf4),   # Feature selection
    ('trf5', trf5)    # Model training
])
```

Training the pipeline:

```python
pip.fit(X_train, y_train)
```

Making predictions:

```python
y_pred = pip.predict(X_test)
```

---

# 🔍 Hyperparameter Tuning with Pipeline

The pipeline also supports **hyperparameter optimization using GridSearchCV**.

Example:

```python
params = {
    'trf5__max_depth': [1,2,3,4,5,None]
}

grid = GridSearchCV(pip, params, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)
```

This allows **tuning model parameters while keeping preprocessing inside the pipeline**.

---

# 🚀 Key Benefits of Using Pipelines

### 1️⃣ Prevents Data Leakage
All preprocessing is applied **inside the training process**, preventing leakage from test data.

### 2️⃣ Reproducible Workflow
Every step is executed in the **same order every time**, ensuring consistent results.

### 3️⃣ Clean and Maintainable Code
Instead of writing long preprocessing scripts, pipelines create a **structured workflow**.

### 4️⃣ Easy Model Deployment
The entire pipeline can be **exported as one object** and used in production.

### 5️⃣ Works with Cross Validation
Pipelines integrate seamlessly with:

- Cross Validation  
- GridSearchCV  
- RandomizedSearchCV  

### 6️⃣ Scalable for Production ML Systems
Pipelines are widely used in **production machine learning systems**.

---

# 📦 Exporting the Pipeline

The trained pipeline can be saved using Pickle:

```python
import pickle

pickle.dump(pip, open("pipeline.pkl","wb"))
```

This allows the **entire workflow to be reused later** without rebuilding preprocessing steps.

---

# 🛠️ Technologies Used

- Python  
- Scikit-Learn  
- Pandas  
- NumPy  
- Jupyter Notebook  

---

# 📁 Project Structure

```
ml-pipeline-project
│
├── ml_pipeline.ipynb
├── titanic-Dataset.csv
├── pipeline.pkl
└── README.md
```

---

# 🎯 Concepts Demonstrated

This project demonstrates key ML engineering concepts:

- Scikit-Learn Pipelines  
- ColumnTransformer  
- Feature Engineering  
- Model Training  
- Cross Validation  
- Hyperparameter Tuning  
- Model Serialization  

---

# 👨‍💻 Author

**Ashish Kathane**  
AI & Machine Learning Student  
JD College of Engineering
