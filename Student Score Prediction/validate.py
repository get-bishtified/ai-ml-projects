from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer
import sagemaker

ENDPOINT_NAME = "student-score-endpoint"

sess = sagemaker.Session()

predictor = Predictor(
    endpoint_name=ENDPOINT_NAME,
    sagemaker_session=sess
)

predictor.serializer = CSVSerializer()
predictor.deserializer = JSONDeserializer()

# Single prediction
print(predictor.predict([5, 80]))

# Batch prediction in one call
print(predictor.predict([
    [5, 80],
    [2, 60],
    [8, 95]
]))
