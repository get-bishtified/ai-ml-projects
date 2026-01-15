import pandas as pd
import sagemaker

sess = sagemaker.Session()
bucket = sess.default_bucket()

df = pd.read_csv("students.csv")

# Reorder for XGBoost: label first
df = df[["score", "hours", "attendance"]]

df.to_csv("/tmp/train.csv", index=False, header=False)

sess.upload_data("/tmp/train.csv", bucket=bucket, key_prefix="student-data")

print("Uploaded to:", f"s3://{bucket}/student-data")
