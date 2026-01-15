import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput

sess = sagemaker.Session()
role = sagemaker.get_execution_role()
bucket = sess.default_bucket()

xgb = Estimator(
    image_uri=sagemaker.image_uris.retrieve(
        "xgboost", sess.boto_region_name, "1.5-1"
    ),
    role=role,
    instance_count=1,
    instance_type="ml.m5.large",
    output_path=f"s3://{bucket}/output",
    sagemaker_session=sess
)

xgb.set_hyperparameters(
    objective="reg:squarederror",
    num_round=50
)

train_input = TrainingInput(
    s3_data=f"s3://{bucket}/student-data",
    content_type="text/csv"
)

xgb.fit({"train": train_input})

print("Model artifact:", xgb.model_data)
