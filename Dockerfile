FROM nvidia/cuda:11.0-base-ubuntu18.04

RUN apt-get update
RUN apt-get install -y python3 python3-pip wget

WORKDIR /work

ADD requirements.txt /work
RUN pip3 install install -r requirements.txt

COPY main.py get_stats.py load_data.py model.py train.py test.py utils.py download_weights.sh /work/
RUN /bin/bash download_weights.sh

ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs
