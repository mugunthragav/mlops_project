# Base Image 
FROM python:3.8-slim 
# Set environment variables 
ENV PYTHONUNBUFFERED=TRUE 
ENV APP_HOME=/app 
WORKDIR $APP_HOME 
# Install dependencies 
COPY requirements.txt . 
RUN pip install -r requirements.txt 
# Copy your code 
COPY . . 
# Expose the serving port 
EXPOSE 5000 
# Serve the model using MLflow 
CMD ["mlflow", "models", "serve", "-m", "mlruns/0/<run_id>/artifacts/ElasticNet_Wine_Model", "-h", "0.0.0.0", "-p", "5000"] 
