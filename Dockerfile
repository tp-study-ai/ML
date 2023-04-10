FROM python:3.10-slim

COPY ./ /app
WORKDIR /app
RUN pip install -r /app/requirements.txt --no-cache-dir
RUN python /app/src/data/get_data.py
CMD python app.py
