import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from joblib import dump
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# Updated dataset paths with generic names
logistic_rf_data_path = 'data/customer_data.csv'
naive_bayes_data_path = 'data/sentiment_data.csv'
kmeans_data_path = 'data/segmentation_data.csv'

# Load Datasets
logistic_rf_data = pd.read_csv(logistic_rf_data_path)
naive_bayes_data = pd.read_csv(naive_bayes_data_path)
kmeans_data = pd.read_csv(kmeans_data_path)

# Prepare Data for Logistic Regression and Random Forest
X = logistic_rf_data.drop(columns=['Target'])  # Features
y = logistic_rf_data['Target']  # Target variable

# Handle imbalance in the dataset using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Scale Data for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save feature names
pd.DataFrame({'features': X.columns}).to_csv('models/feature_names.csv', index=False)

# Logistic Regression
logistic_model = LogisticRegression(random_state=42, max_iter=1000)
logistic_model.fit(pd.DataFrame(X_train_scaled, columns=X.columns), y_train)  # Train with feature names
dump(logistic_model, 'models/logistic_model.joblib')
dump(scaler, 'models/scaler.joblib')
print("Logistic Regression Model Trained.")

# Evaluate Logistic Regression on Test Data
y_pred_logistic = logistic_model.predict(X_test_scaled)
print("Classification Report for Logistic Regression:")
print(classification_report(y_test, y_pred_logistic))

# Random Forest
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest.fit(X_train, y_train)
dump(random_forest, 'models/random_forest_model.joblib')
print("Random Forest Model Trained.")

# Evaluate Random Forest on Test Data
y_pred_rf = random_forest.predict(X_test)
print("Classification Report for Random Forest:")
print(classification_report(y_test, y_pred_rf))

# Prepare Data for Naive Bayes
try:
    # Check if the 'Comment' column exists
    if 'Comment' not in naive_bayes_data.columns:
        raise KeyError("The 'Comment' column is missing in the sentiment dataset.")
    
    # Check if the 'Sentiment' column exists
    if 'Sentiment' not in naive_bayes_data.columns:
        raise KeyError("The 'Sentiment' column is missing in the sentiment dataset.")
    
    # Vectorize the 'Comment' column
    vectorizer = CountVectorizer()
    X_sentiment = vectorizer.fit_transform(naive_bayes_data['Comment'])
    y_sentiment = naive_bayes_data['Sentiment']

    # Train-Test Split
    X_train_sent, X_test_sent, y_train_sent, y_test_sent = train_test_split(
        X_sentiment, y_sentiment, test_size=0.2, random_state=42
    )

    # Train the Naive Bayes Model
    naive_bayes_model = GaussianNB()
    naive_bayes_model.fit(X_train_sent.toarray(), y_train_sent)

    # Save the Model and Vectorizer
    dump(naive_bayes_model, 'models/naive_bayes_model.joblib')
    dump(vectorizer, 'models/naive_bayes_vectorizer.joblib')
    print("Naive Bayes Model Trained.")

    # Evaluate Naive Bayes on Test Data
    y_pred_nb = naive_bayes_model.predict(X_test_sent.toarray())
    print("Classification Report for Naive Bayes:")
    print(classification_report(y_test_sent, y_pred_nb))

except KeyError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# K-Means Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(kmeans_data)
dump(kmeans, 'models/kmeans_model.joblib')
print("K-Means Model Trained.")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Collect model performance metrics
metrics = {
    "Logistic Regression": {
        "Accuracy": accuracy_score(y_test, y_pred_logistic),
        "Precision": precision_score(y_test, y_pred_logistic),
        "Recall": recall_score(y_test, y_pred_logistic),
        "F1-Score": f1_score(y_test, y_pred_logistic)
    },
    "Random Forest": {
        "Accuracy": accuracy_score(y_test, y_pred_rf),
        "Precision": precision_score(y_test, y_pred_rf),
        "Recall": recall_score(y_test, y_pred_rf),
        "F1-Score": f1_score(y_test, y_pred_rf)
    },
    "Naive Bayes": {
        "Accuracy": accuracy_score(y_test_sent, y_pred_nb),
        "Precision": precision_score(y_test_sent, y_pred_nb),
        "Recall": recall_score(y_test_sent, y_pred_nb),
        "F1-Score": f1_score(y_test_sent, y_pred_nb)
    }
}

# Convert metrics to DataFrame
metrics_df = pd.DataFrame.from_dict(metrics, orient="index")
metrics_df.to_csv('models/model_metrics.csv')
print("Model metrics saved to 'models/model_metrics.csv'.")

# Save Feature Importance for Random Forest
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': random_forest.feature_importances_
}).sort_values(by='Importance', ascending=False)

feature_importance.to_csv('models/random_forest_feature_importance.csv', index=False)
print("Random Forest Feature Importance saved to 'models/random_forest_feature_importance.csv'.")
