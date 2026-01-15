from flask import Flask, request, render_template
import boto3

app = Flask(__name__)

# Replace with your real endpoint name
ENDPOINT = "student-score-endpoint"

# Make sure region matches where your endpoint is deployed
runtime = boto3.client("sagemaker-runtime", region_name="ap-south-1")

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    error = None

    if request.method == "POST":
        try:
            hours = float(request.form["hours"])
            attendance = float(request.form["attendance"])

            # XGBoost expects plain CSV: "5,80"
            payload = f"{hours},{attendance}"
            print("Sending payload:", payload)

            response = runtime.invoke_endpoint(
                EndpointName=ENDPOINT,
                ContentType="text/csv",
                Body=payload
            )

            result = response["Body"].read().decode()

        except Exception as e:
            error = str(e)
            print("Error:", error)

    return render_template("index.html", result=result, error=error)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
