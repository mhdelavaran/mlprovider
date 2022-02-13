FROM python:3.7

# layer caching for faster builds
COPY requirements.txt /
RUN pip3 install  --verbose  -r /requirements.txt

#COPY app.py /app.py
ADD . /app
WORKDIR /app

ENV FLASK_ENV=development

CMD flask run --host=0.0.0.0

#CMD gunicorn --workers $WORKERS \
#  --threads $THREADS \
#  --bind 0.0.0.0:$PORT_APP \
#  --log-level DEBUG \
#  app:app