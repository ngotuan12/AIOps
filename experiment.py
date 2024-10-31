from churn_data_generator import ChurnDataGenerator
from churn_model_tuner import ChurnModelTuner
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from churn_model_tuner_mlflow import ChurnModelTunerMLFlow

# Sử dụng
if __name__ == "__main__":
    # Tạo dữ liệu
    generator = ChurnDataGenerator(n_samples=100)
    df = generator.generate_data()
    # Chia và chuẩn hóa dữ liệu
    X = df.drop('churn', axis=1)
    y = df['churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # hyper params
    param_distributions = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'class_weight': ['balanced', 'balanced_subsample']
    }
    # grid search
    tuner = ChurnModelTuner(X_train_scaled, X_test_scaled, y_train, y_test, param_distributions)
    tuner_results = tuner.grid_search_tuning()
    # Ml flow tracking
    flow = ChurnModelTunerMLFlow()
    flow.log_model("grid_search_tuning", tuner_results)
