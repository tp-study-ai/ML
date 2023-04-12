FROM python:3.10-slim

COPY ./ /app
WORKDIR /app
RUN pip install -r /app/requirements.txt --no-cache-dir && pip3 install torch --index-url https://download.pytorch.org/whl/cpu --no-cache-dir
RUN python /app/src/data/get_data.py
EXPOSE 9000
CMD python app.py
