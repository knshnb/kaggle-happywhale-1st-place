FROM nvcr.io/nvidia/pytorch:22.02-py3

COPY requirements.txt .
RUN pip install -r requirements.txt
