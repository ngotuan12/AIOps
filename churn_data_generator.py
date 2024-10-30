import pandas as pd
from sklearn.datasets import make_classification


class ChurnDataGenerator:
    def __init__(self, n_samples=1000, n_features=10, n_informative=6, n_redundant=2):
        """
        Khởi tạo generator với các tham số

        Parameters:
        -----------
        n_samples : int
            Số lượng mẫu cần tạo
        n_features : int
            Tổng số đặc trưng
        n_informative : int
            Số đặc trưng có tính thông tin
        n_redundant : int
            Số đặc trưng dư thừa
        """
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_informative = n_informative
        self.n_redundant = n_redundant

        # Tên các đặc trưng
        self.feature_names = [
            'tenure',
            'monthly_charges',
            'total_charges',
            'usage_level',
            'support_calls',
            'service_rating',
            'payment_delay',
            'contract_type',
            'service_additions',
            'customer_age'
        ]

    def generate_data(self):
        """
        Tạo dữ liệu mẫu sử dụng make_classification
        """
        # Tạo dữ liệu với make_classification
        X, y = make_classification(
            n_samples=self.n_samples,
            n_features=self.n_features,
            n_informative=self.n_informative,
            n_redundant=self.n_redundant,
            n_clusters_per_class=2,
            class_sep=0.8,
            random_state=42,
            weights=[0.7, 0.3]  # Tỷ lệ không cân bằng giữa các class
        )

        # Chuyển đổi thành DataFrame với tên các đặc trưng có ý nghĩa
        df = pd.DataFrame(X, columns=self.feature_names)

        # Thêm cột target
        df['churn'] = y

        # Chuẩn hóa và điều chỉnh dữ liệu để có ý nghĩa thực tế hơn
        self._transform_features(df)

        return df

    @staticmethod
    def _transform_features(df):
        """
        Biến đổi các đặc trưng để có ý nghĩa thực tế hơn
        """
        # Tenure: số tháng (1-72 tháng)
        df['tenure'] = (df['tenure'] - df['tenure'].min()) / (df['tenure'].max() - df['tenure'].min()) * 71 + 1
        df['tenure'] = df['tenure'].round()

        # Monthly charges: $20-$200
        df['monthly_charges'] = (df['monthly_charges'] - df['monthly_charges'].min()) / \
                                (df['monthly_charges'].max() - df['monthly_charges'].min()) * 180 + 20

        # Total charges: dựa trên tenure và monthly charges
        df['total_charges'] = df['tenure'] * df['monthly_charges']

        # Usage level: 0-1000 units
        df['usage_level'] = (df['usage_level'] - df['usage_level'].min()) / \
                            (df['usage_level'].max() - df['usage_level'].min()) * 1000

        # Support calls: 0-10 calls
        df['support_calls'] = (df['support_calls'] - df['support_calls'].min()) / \
                              (df['support_calls'].max() - df['support_calls'].min()) * 10
        df['support_calls'] = df['support_calls'].round()

        # Service rating: 1-5
        df['service_rating'] = (df['service_rating'] - df['service_rating'].min()) / \
                               (df['service_rating'].max() - df['service_rating'].min()) * 4 + 1
        df['service_rating'] = df['service_rating'].round()

        # Payment delay: 0-30 days
        df['payment_delay'] = (df['payment_delay'] - df['payment_delay'].min()) / \
                              (df['payment_delay'].max() - df['payment_delay'].min()) * 30
        df['payment_delay'] = df['payment_delay'].round()

        # Contract type: 0 (Month-to-month), 1 (One year), 2 (Two year)
        df['contract_type'] = pd.qcut(df['contract_type'], q=3, labels=[0, 1, 2])

        # Service additions: 0-5 additional services
        df['service_additions'] = (df['service_additions'] - df['service_additions'].min()) / \
                                  (df['service_additions'].max() - df['service_additions'].min()) * 5
        df['service_additions'] = df['service_additions'].round()

        # Customer age: 18-80 years
        df['customer_age'] = (df['customer_age'] - df['customer_age'].min()) / \
                             (df['customer_age'].max() - df['customer_age'].min()) * 62 + 18
        df['customer_age'] = df['customer_age'].round()
