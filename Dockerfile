FROM python:3.11-slim
ENV PYTHONUNBUFFERED 1
ENV APP_HOME /usr/src/app

RUN mkdir -p $APP_HOME
WORKDIR $APP_HOME

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8051

CMD ["gunicorn", "--bind", "0.0.0.0:8051", "app:server"]

