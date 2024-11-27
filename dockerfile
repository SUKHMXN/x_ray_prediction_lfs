# 1. Start with the base image: TensorFlow with Python support
FROM tensorflow/tensorflow:2.10.0-py3

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the current directory contents into the container at /app
COPY . .

# 4. Install Python dependencies (from the requirements.txt file)
RUN pip install --no-cache-dir -r requirements.txt

# 5. Expose port 8080 (Flask runs on port 5000 by default, you can adjust it)
EXPOSE 8080

# 6. Set the environment variable for Google credentials (optional, for GCS access)
ENV GOOGLE_APPLICATION_CREDENTIALS /app/service-account-key.json

# 7. Start the Flask app
CMD ["python", "app.py"]
