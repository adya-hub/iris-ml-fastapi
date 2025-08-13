# Step 1: Use an official Python image
FROM python:3.11

# Step 2: Set working directory inside container
WORKDIR /app

# Step 3: Copy dependencies first
COPY requirements.txt .

# Step 4: Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy the rest of the files
COPY iris_api.py .
COPY iris_model.pkl .

# Step 6: Expose the FastAPI port
EXPOSE 8000

# Step 7: Command to run FastAPI app
CMD ["uvicorn", "iris_api:app", "--host", "0.0.0.0", "--port", "8000"]
