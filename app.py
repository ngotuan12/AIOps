from flask import Flask
import uvicorn
import json
from churn_model_tuner_mlflow import ChurnModelTunerMLFlow
from asgiref.wsgi import WsgiToAsgi

app = Flask(__name__)

# Convert WSGI app to ASGI
asgi_app = WsgiToAsgi(app)


@app.route('/')
def index():
    # Ml flow tracking
    flow = ChurnModelTunerMLFlow()
    # load best run
    best_run = flow.load_best_model()
    best_params = json.dumps(best_run['params.best_params'])
    param_distributions = json.dumps(best_run['params.param_distributions'])
    accuracy = best_run['metrics.accuracy']
    roc_auc = best_run['metrics.roc_auc']
    # customer test data
    customer_df, customer_data = flow.create_customer(
        tenure=12,  # Thời gian sử dụng: 12 tháng
        monthly_charges=80,  # Phí hàng tháng: $80
        usage_level=500,  # Mức độ sử dụng: 500 units
        support_calls=3,  # 3 cuộc gọi hỗ trợ
        service_rating=4,  # Đánh giá 4/5 sao
        payment_delay=5,  # Trễ thanh toán 5 ngày
        contract_type=0,  # Hợp đồng theo tháng
        service_additions=2,  # 2 dịch vụ bổ sung
        customer_age=35  # 35 tuổi
    )
    # prediction
    prediction_result = flow.predict_single_customer(customer_df)
    print(f"Dự đoán: {'Rời bỏ' if prediction_result['prediction'] == 1 else 'Không rời bỏ'}")
    print(f"Xác suất rời bỏ: {prediction_result['churn_probability']:.2%}")
    print(f"Mức độ rủi ro: {prediction_result['risk_level']}")
    # html
    html_content = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>My Flask App</title>
            </head>
            <body>
                <h1>AI Ops, ML ops, Data Ops final project!</h1>
                <p>Best param: {best_params}</p>
                <p>param_distributions: {param_distributions}</p>
                <p>accuracy: {accuracy}</p>
                <p>roc_auc: {roc_auc}</p>
                <p>customer data : {json.dumps(customer_data)}</p>
                <p>prediction result : {json.dumps(prediction_result)}</p>
                <p>Dự đoán: {'Rời bỏ' if prediction_result['prediction'] == 1 else 'Không rời bỏ'}</p>
                <p>Xác suất rời bỏ:{prediction_result['churn_probability']:.2%}</p>
                <p>Mức độ rủi ro: {prediction_result['risk_level']}</p>
            </body>
            </html>
            """
    return html_content


@app.route('/predict')
def predict():
    # Ml flow tracking
    flow = ChurnModelTunerMLFlow()
    # load best run
    best_run = flow.load_best_model()
    best_params = json.dumps(best_run['params.best_params'])
    param_distributions = json.dumps(best_run['params.param_distributions'])
    accuracy = best_run['metrics.accuracy']
    roc_auc = best_run['metrics.roc_auc']
    # customer test data
    customer_df, customer_data = flow.create_customer(
        tenure=12,  # Thời gian sử dụng: 12 tháng
        monthly_charges=80,  # Phí hàng tháng: $80
        usage_level=500,  # Mức độ sử dụng: 500 units
        support_calls=3,  # 3 cuộc gọi hỗ trợ
        service_rating=4,  # Đánh giá 4/5 sao
        payment_delay=5,  # Trễ thanh toán 5 ngày
        contract_type=0,  # Hợp đồng theo tháng
        service_additions=2,  # 2 dịch vụ bổ sung
        customer_age=35  # 35 tuổi
    )
    # prediction
    prediction_result = flow.predict_single_customer(customer_df)
    print(f"Dự đoán: {'Rời bỏ' if prediction_result['prediction'] == 1 else 'Không rời bỏ'}")
    print(f"Xác suất rời bỏ: {prediction_result['churn_probability']:.2%}")
    print(f"Mức độ rủi ro: {prediction_result['risk_level']}")
    # html
    html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>My Flask App</title>
        </head>
        <body>
            <h1>AI Ops, ML ops, Data Ops final project!</h1>
            <p>Best param: {best_params}</p>
            <p>param_distributions: {param_distributions}</p>
            <p>accuracy: {accuracy}</p>
            <p>roc_auc: {roc_auc}</p>
            <p>customer data : {json.dumps(customer_data)}</p>
            <p>prediction result : {json.dumps(prediction_result)}</p>
            <p>Dự đoán: {'Rời bỏ' if prediction_result['prediction'] == 1 else 'Không rời bỏ'}</p>
            <p>Xác suất rời bỏ:{prediction_result['churn_probability']:.2%}</p>
            <p>Mức độ rủi ro: {prediction_result['risk_level']}</p>
        </body>
        </html>
        """
    return html_content


if __name__ == '__main__':
    uvicorn.run(asgi_app, host="127.0.0.1", port=1978)
