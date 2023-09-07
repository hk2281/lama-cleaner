FROM python:3.10.8 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    libsm6 libxext6 ffmpeg libfontconfig1 libxrender1 libgl1-mesa-glx \
    curl gcc build-essential git bash

RUN pip install --upgrade pip && \
    pip install torch==1.13.1 torchvision==0.14.1 --extra-index-url https://download.pytorch.org/whl/cpu


RUN git clone https://github.com/hk2281/lama-cleaner.git

RUN cd lama-cleaner
RUN ls 
RUN pwd
RUN pip install lama-cleaner/.

FROM builder AS final
WORKDIR lama-cleaner
CMD ["python","main.py"]