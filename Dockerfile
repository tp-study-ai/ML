FROM pytorch/pytorch:latest

COPY ./ /app
WORKDIR /app
RUN pip install -r /app/requirements.txt --no-cache-dir
RUN python /app/src/data/get_data.py
EXPOSE 9000
CMD python app.py
