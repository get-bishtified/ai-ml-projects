import sagemaker
from sagemaker.model import Model
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer
from sagemaker.predictor import Predictor

sess = sagemaker.Session()
role = sagemaker.get_execution_role()
region = sess.boto_region_name

MODEL_DATA = "s3://sagemaker-ap-south-1-638900790108/output/sagemaker-xgboost-2026-01-15-11-23-30-332/output/model.tar.gz"
ENDPOINT_NAME = "student-score-endpoint"

model = Model(
    image_uri=sagemaker.image_uris.retrieve("xgboost", region, "1.5-1"),
    model_data=MODEL_DATA,
    role=role,
    sagemaker_session=sess,
)

# Deploy (may return None in script mode)
model.deploy(
    instance_type="ml.t2.medium",
    initial_instance_count=1,
    endpoint_name=ENDPOINT_NAME,
    wait=True
)

# Always create Predictor explicitly
predictor = Predictor(
    endpoint_name=ENDPOINT_NAME,
    sagemaker_session=sess
)

predictor.serializer = CSVSerializer()
predictor.deserializer = JSONDeserializer()

print("Endpoint deployed and predictor ready:", ENDPOINT_NAME)
