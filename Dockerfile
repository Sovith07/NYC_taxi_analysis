ARG PYTHON_VERSION=3.12.10
FROM python:${PYTHON_VERSION}-slim as base



# Set the working directory to /app
WORKDIR /app

# Copy the required files and directory into the container at /app
COPY service_uvicorn.py /app/service_uvicorn.py
COPY models/model.joblib /app/model.joblib
COPY src/ /app/src/
COPY dev-requirements.txt /app/dev-requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install -r dev-requirements.txt

# Copy files from S3 inside docker
# RUN mkdir /app/models
# RUN aws s3 cp s3://creditcard-project/models/model.joblib /app/models/model.joblib

# Run app.py when the container launches
CMD ["python", "service_uvicorn.py"]
