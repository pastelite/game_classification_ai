# Start from the official Ubuntu base image
FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime

# Set the current working directory to /home/app
WORKDIR /home/app

# Project files from the local directory to the image
COPY . /home/app

# Install Python packages
RUN pip --no-cache-dir install --upgrade pip
RUN pip --no-cache-dir install -r /home/app/requirements.txt

# Start FastAPI & gradio
CMD ["uvicorn", "fastapi_start:app", "--host", "0.0.0.0"]

# port
EXPOSE 8000
