from flask import Flask
import json
from churn_model_tuner_mlflow import ChurnModelTunerMLFlow
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)


@app.route('/predict')
def predict():
    # Ml flow tracking
    flow = ChurnModelTunerMLFlow()
    best_run = flow.load_best_model()
    best_params = json.dumps(best_run['params.best_params'])
    param_distributions = json.dumps(best_run['params.param_distributions'])
    accuracy = best_run['metrics.accuracy']
    roc_auc = best_run['metrics.roc_auc']
    customer_df,customer_data = flow.create_customer()
    prediction_result = flow.predict_single_customer(customer_df)
    print(f"Dự đoán: {'Rời bỏ' if prediction_result['prediction'] == 1 else 'Không rời bỏ'}")
    print(f"Xác suất rời bỏ: {prediction_result['churn_probability']:.2%}")
    print(f"Mức độ rủi ro: {prediction_result['risk_level']}")
    html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>My Flask App</title>
        </head>
        <body>
            <h1>Welcome to My Flask App!</h1>
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
    app.run(host='0.0.0.0', port=1978)
