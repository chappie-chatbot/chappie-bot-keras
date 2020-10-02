FROM tensorflow/tensorflow

ADD ./ /app

WORKDIR /app

RUN pip install -r requirements.txt

RUN python train.py

ENV PORT 80

ENTRYPOINT ["/usr/local/bin/python", "flaskserver.py"]
