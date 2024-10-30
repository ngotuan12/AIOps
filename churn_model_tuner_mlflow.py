import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
import mlflow
import mlflow.sklearn
from sklearn.metrics import classification_report, confusion_matrix


class ChurnModelTunerMLFlow:
    def __init__(self, experiment_name="Churn Prediction"):
        # Thiết lập MLflow
        mlflow.set_tracking_uri("http://127.0.0.1:7000")
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name
        self.model = None
        self.run_id = None
        self.experiment_id = None

    @staticmethod
    def log_model(run_name, tuner_results):
        # grid search tunner
        with mlflow.start_run(run_name=run_name):
            # Log parameters
            mlflow.log_params({"search_type": tuner_results["search_type"]})
            mlflow.log_params({"param_distributions": str(tuner_results["param_distributions"])})
            # Log best parameters
            mlflow.log_params({"best_params": tuner_results['best_params']})
            # Log all metrics
            mlflow.log_metrics(tuner_results['metrics'])
            # Log model
            mlflow.sklearn.log_model(tuner_results['best_model'], "model")

    def load_best_model(self):
        """
        Load model tốt nhất từ MLflow dựa trên metric

        Parameters:
        -----------
        experiment_name : str
            Tên của experiment
        metric : str
            Metric để chọn model tốt nhất
        """
        # Lấy experiment
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            raise ValueError(f"Không tìm thấy experiment '{self.experiment_name}'")

        self.experiment_id = experiment.experiment_id

        # Lấy tất cả runs trong experiment
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            order_by=[f"metrics.roc_auc DESC", f"metrics.accuracy DESC"]
        )

        if runs.empty:
            raise ValueError(f"Không tìm thấy runs trong experiment '{self.experiment_name}'")

        # Lấy run tốt nhất
        best_run = runs.iloc[0]
        self.run_id = best_run.run_id

        print(f"Loading model từ run: {self.run_id}")
        print(f"Metrics của model:")
        for col in runs.columns:
            if col.startswith('metrics.'):
                metric_name = col.replace('metrics.', '')
                print(f"{metric_name}: {best_run[col]:.4f}")
        # Load model
        self.model = mlflow.sklearn.load_model(f"runs:/{self.run_id}/model")
        print("\nĐã load model thành công!")
        return best_run

    def predict_single_customer(self, customer_data: np.ndarray) -> Dict[str, Any]:
        """
        Dự đoán cho một khách hàng cụ thể

        Parameters:
        -----------
        customer_data : array-like
            Dữ liệu của một khách hàng

        Returns:
        --------
        dict
            Kết quả dự đoán và xác suất
        """
        if self.model is None:
            raise ValueError("Chưa load model! Hãy gọi load_best_model() trước.")

        # Reshape data nếu cần
        if customer_data.ndim == 1:
            customer_data = customer_data.reshape(1, -1)

        # Dự đoán
        prediction = self.model.predict(customer_data)[0]
        probability = self.model.predict_proba(customer_data)[0, 1]

        result = {
            'prediction': int(prediction),
            'churn_probability': float(probability),
            'risk_level': 'Cao' if probability > 0.7 else 'Trung bình' if probability > 0.3 else 'Thấp'
        }

        return result

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Thực hiện dự đoán với model đã load

        Returns:
        --------
        predictions : array
            Nhãn dự đoán
        probabilities : array
            Xác suất dự đoán
        """
        if self.model is None:
            raise ValueError("Chưa load model! Hãy gọi load_best_model() trước.")

        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]

        return predictions, probabilities

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Đánh giá hiệu suất của model

        Returns:
        --------
        dict
            Dictionary chứa các metrics và visualizations
        """
        if self.model is None:
            raise ValueError("Chưa load model! Hãy gọi load_best_model() trước.")

        # Thực hiện dự đoán
        y_pred, y_prob = self.predict(X)

        # Tính toán metrics
        results = {
            'classification_report': classification_report(y, y_pred),
            'confusion_matrix': confusion_matrix(y, y_pred),
            'predictions': y_pred,
            'probabilities': y_prob
        }

        # Feature importance nếu model là RandomForest
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': [f'feature_{i}' for i in range(len(self.model.feature_importances_))],
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            results['feature_importance'] = feature_importance

        return results

    def create_customer(self,
                        tenure=12,  # Số tháng sử dụng dịch vụ
                        monthly_charges=80,  # Phí hàng tháng ($)
                        usage_level=500,  # Mức độ sử dụng (0-1000)
                        support_calls=3,  # Số cuộc gọi hỗ trợ (0-10)
                        service_rating=4,  # Đánh giá dịch vụ (1-5)
                        payment_delay=5,  # Độ trễ thanh toán (ngày)
                        contract_type=0,  # Loại hợp đồng (0: Monthly, 1: One year, 2: Two year)
                        service_additions=2,  # Số dịch vụ bổ sung (0-5)
                        customer_age=35):  # Tuổi khách hàng
        """
        Tạo dữ liệu cho một khách hàng mới
        """
        # Tính total charges
        total_charges = tenure * monthly_charges

        # Tạo dictionary chứa thông tin khách hàng
        customer_data = {
            'tenure': tenure,
            'monthly_charges': monthly_charges,
            'total_charges': total_charges,
            'usage_level': usage_level,
            'support_calls': support_calls,
            'service_rating': service_rating,
            'payment_delay': payment_delay,
            'contract_type': contract_type,
            'service_additions': service_additions,
            'customer_age': customer_age
        }

        # Tạo DataFrame
        customer_df = pd.DataFrame([customer_data])

        return customer_df, customer_data
