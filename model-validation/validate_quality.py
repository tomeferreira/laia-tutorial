import mlflow
import os
import sys

MIN_ACCURACY = 0.9

# Connect to MLflow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

# Get the model version by alias
client = mlflow.MlflowClient()
model_versions = client.get_model_version_by_alias(
    name=os.getenv("MLFLOW_MODEL_NAME"),
    alias=os.getenv("MODEL_ALIAS")  # Will be the commit SHA
)

# Get the run that created this model version
run_id = model_versions.run_id
run = client.get_run(run_id)

# Access metrics
accuracy = run.data.metrics.get("accuracy")

if accuracy < MIN_ACCURACY:
	print("❌ Model FAILED quality gate")
	print(f"    Accuracy = {str(accuracy)} (threshold{str(MIN_ACCURACY)})")
	print("Pipeline stopped. Model will NOT be promoted.")
	sys.exit(1)

print("✅ Model passed quality gate")
print(f"    Accuracy = {str(accuracy)} (threshold{str(MIN_ACCURACY)})")