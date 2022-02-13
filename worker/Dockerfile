FROM python:3.7

COPY requirements.txt /
RUN pip install  --verbose -r /requirements.txt

ADD . /worker
WORKDIR /worker

ENTRYPOINT celery -A tasks worker --loglevel=info
