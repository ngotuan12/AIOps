from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, classification_report
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, \
    confusion_matrix


class ChurnModelTuner:
    def __init__(self, X_train, X_test, y_train, y_test, param_distributions):
        """
        Khởi tạo tuner với dữ liệu đã được chuẩn hóa
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.best_model = None
        self.best_params = None
        self.cv_results = None
        self.param_distributions = param_distributions

    def grid_search_tuning(self, verbose=2):
        """
        Thực hiện Grid Search với các tham số đã định nghĩa sẵn
        """
        # Định nghĩa grid tham số
        # param_distributions = {
        #     'n_estimators': [100, 200, 300],
        #     'max_depth': [10, 20, 30, None],
        #     'min_samples_split': [2, 5, 10],
        #     'min_samples_leaf': [1, 2, 4],
        #     'max_features': ['sqrt', 'log2', None],
        #     'class_weight': ['balanced', 'balanced_subsample']
        # }

        # Khởi tạo model
        base_model = RandomForestClassifier(random_state=42)

        # Định nghĩa scorer
        scorer = make_scorer(roc_auc_score)

        # Thực hiện Grid Search
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=self.param_distributions,
            cv=5,
            scoring=scorer,
            n_jobs=-1,
            verbose=verbose
        )

        print("Bắt đầu Grid Search...")
        grid_search.fit(self.X_train, self.y_train)

        # Lưu kết quả
        self.best_model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.cv_results = pd.DataFrame(grid_search.cv_results_)

        return self._evaluate_model("Grid Search")

    def random_search_tuning(self, n_iter=100, verbose=2):
        """
        Thực hiện Random Search với phạm vi tham số rộng hơn
        """
        # Định nghĩa phân phối tham số
        # param_distributions = {
        #     'n_estimators': np.arange(100, 500, 50),
        #     'max_depth': np.arange(10, 50, 5),
        #     'min_samples_split': np.arange(2, 20, 2),
        #     'min_samples_leaf': np.arange(1, 10),
        #     'max_features': ['sqrt', 'log2', None],
        #     'class_weight': ['balanced', 'balanced_subsample'],
        #     'criterion': ['gini', 'entropy']
        # }

        # Khởi tạo model
        base_model = RandomForestClassifier(random_state=42)

        # Định nghĩa scorer
        scorer = make_scorer(roc_auc_score)

        # Thực hiện Random Search
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=self.param_distributions,
            n_iter=n_iter,
            cv=5,
            scoring=scorer,
            n_jobs=-1,
            verbose=verbose,
            random_state=42
        )

        print("Bắt đầu Random Search...")
        random_search.fit(self.X_train, self.y_train)

        # Lưu kết quả
        self.best_model = random_search.best_estimator_
        self.best_params = random_search.best_params_
        self.cv_results = pd.DataFrame(random_search.cv_results_)

        return self._evaluate_model("Random Search")

    def _evaluate_model(self, search_type):
        """
        Đánh giá mô hình tốt nhất
        """
        # Dự đoán trên tập test
        y_pred = self.best_model.predict(self.X_test)
        y_pred_proba = self.best_model.predict_proba(self.X_test)[:, 1]
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred),
            'recall': recall_score(self.y_test, y_pred),
            'f1': f1_score(self.y_test, y_pred),
            'roc_auc': roc_auc_score(self.y_test, y_pred_proba),
        }
        # Tính các metrics
        results = {
            'metrics': metrics,
            'search_type': search_type,
            'best_params': self.best_params,
            'best_model': self.best_model,
            'param_distributions': self.param_distributions,
            'classification_report': classification_report(self.y_test, y_pred),
            'roc_auc_score': roc_auc_score(self.y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(self.y_test, y_pred),
            'feature_importance': pd.DataFrame({
                'feature': [f'feature_{i}' for i in range(self.X_train.shape[1])],
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
        }

        return results
