FROM tensorflow/tensorflow

ADD ./ /app

WORKDIR /app

RUN pip install -r requirements.txt

RUN python train.py

ENV PORT 8082

ENTRYPOINT ["/usr/local/bin/python", "flaskserver.py"]
