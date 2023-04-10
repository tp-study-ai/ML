FROM python:3.10-slim

COPY . /home/app/
WORKDIR /home/app
RUN pip install --no-cache-dir -r requirements.txt && python src/data/get_data.py
CMD ["python", "app.py"]
