from main import model_training_flow
from prefect import serve

if __name__ == "__main__":
    train_model_deployment = model_training_flow.to_deployment(
        name="Model training Deployment",
        version="0.1.0",
        tags=["training", "model"],
        cron="0 0 * * *",
        parameters={
            "trainset_path": "abalone/abalone.data",
            "save_model_path": "src/web_service/local_objects/model.pkl",
        },
    )
    serve(train_model_deployment)
