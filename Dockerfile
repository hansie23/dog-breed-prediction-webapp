FROM python:3.9.12
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt;heroku ps:scale web=1
EXPOSE $PORT
CMD gunicorn --workers=4 --bind 0.0.0.0:$PORT app:app
