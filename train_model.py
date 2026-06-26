import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Load Dataset
df = pd.read_csv('data/online_shoppers_intention.csv')

# Data Cleaning
df = df.drop_duplicates()

# Separate Features and Target
X = df.drop(columns=['Revenue'])
y = df['Revenue'].astype(int)

# Identify Numeric and Categorical Features
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'bool']).columns.tolist()

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create Pipeline with SMOTE and XGBoost
# Note: SMOTE must be in an imblearn Pipeline to be applied only to training data
model_pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    ))
])

# Train Model
print("Training XGBoost model with SMOTE...")
model_pipeline.fit(X_train, y_train)

# Evaluate Model
y_pred = model_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save Model
model_filename = 'models/model_prediksi_pembelian_ecommerce.pkl'
joblib.dump(model_pipeline, model_filename)
print(f"\nModel saved as {model_filename}")
