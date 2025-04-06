# Use Python base image
FROM python:3.10-slim

# Install Java and other dependencies
RUN apt-get update && \
    apt-get install -y default-jre unzip && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy all files into container
COPY . .

# Make the PaDEL script executable
RUN chmod +x padel.sh

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose default Streamlit port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "main.py"]
