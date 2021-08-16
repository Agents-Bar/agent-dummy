FROM python:3.8-slim
# set path to our python api file
ENV MODULE_NAME="app.main"

# RUN pip install jsons
COPY ./setup.* /app/
WORKDIR /app/

RUN pip install --upgrade pip
RUN pip install -e .

COPY ./ /app

LABEL agents-bar-dummy-agent=v0.1.2
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80", "--app-dir", "/app"]
