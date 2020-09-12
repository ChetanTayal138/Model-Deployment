ARG TF_VERSION=2.2.0

FROM tensorflow/tensorflow:${TF_VERSION}

RUN apt-get update -y &&\apt-get install -y python3-pip python3-dev &&\apt-get install -y libsm6 libxext6 libxrender-dev libgl1-mesa-glx

COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip3 install --upgrade pip &&\pip3 install -r requirements.txt

COPY . /app

ENTRYPOINT ["python3"]

CMD ["app.py"]

