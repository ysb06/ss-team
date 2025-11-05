# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

# Expose port 80 for the HTTP server
EXPOSE 80

# Run the server
CMD ["python", "function.py"]
