FROM nvidia/cuda:11.4.2-cudnn8-runtime-ubuntu20.04 as base

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt install -y build-essential && \
    apt-get install --no-install-recommends -y python3.8 python3-pip libpq-dev python3-dev python3-wheel libgtk2.0-dev libgl1-mesa-dev && \
    rm -rf /var/lib/apt/lists/*

RUN python3.8 -m pip install --upgrade pip
RUN pip3 install -U setuptools

COPY ./requirements.txt /requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install --no-cache-dir picsellia

ENV PATH=$PATH:/usr/local/cuda/bin
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/cuda/lib:/usr/local/cuda/lib64

COPY ./training-folder /training-folder

CMD ["/training-folder/launch_subprocess.py"]
ENTRYPOINT ["python3"]
RUN chown -R 42420:42420 /training-folder
