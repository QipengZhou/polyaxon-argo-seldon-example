FROM docker.dm-ai.cn/public/python:3.7-slim

COPY . /app
WORKDIR /app
RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip3 install seldon-core \
    && pip3 install -r requirements.txt
EXPOSE 5000

ENV MODEL_NAME MnistModel
ENV API_TYPE REST
ENV SERVICE_TYPE MODEL
ENV PERSISTENCE 0

CMD exec seldon-core-microservice $MODEL_NAME $API_TYPE \
     --service-type $SERVICE_TYPE \
     --persistence $PERSISTENCE