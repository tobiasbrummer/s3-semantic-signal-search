FROM custom-nvidia-cuda:latest

WORKDIR /workspace

COPY requirements.txt .

RUN pip install -r requirements.txt
