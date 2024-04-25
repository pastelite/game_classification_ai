# Start from the official Ubuntu base image
FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime

# Set the current working directory to /home/app
WORKDIR /home/app

# Uncomment and fill in your account if you are working under MU network
# ENV http_proxy "http://<USER>:<PASSWD>@proxy-sa.mahidol:8080"
# ENV https_proxy "http://<USER>:<PASSWD>@proxy-sa.mahidol:8080"
# ENV ftp_proxy "http://<USER>:<PASSWD>@proxy-sa.mahidol:8080"
# ENV no_proxy "localhost,127.0.0.1,::1"

# Project files from the local directory to the image
COPY . /home/app

# Update software in Ubuntu
# RUN apt-get update && apt-get install -y --no-install-recommends \
#         build-essential \
#         wget \
#         git \
#         vim \
#         bzip2 \
#         ca-certificates \
#         libsm6 \
#         libxext6

# Setup Miniconda
# ENV PATH /opt/conda/bin:$PATH
# RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
#     /bin/bash ~/miniconda.sh -b -p /opt/conda && \
#     rm ~/miniconda.sh && \
#     ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
#     echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
#     echo "conda activate base" >> ~/.bashrc

# Install Python packages
RUN pip --no-cache-dir install --upgrade pip
RUN pip --no-cache-dir install -r /home/app/requirements.txt

# Start FastAPI & gradio
CMD ["/bin/bash","-c","python gradio_start.py & uvicorn fastapi_start:app --host 0.0.0.0"]

# port
EXPOSE 8000
EXPOSE 7860
