# pull base image
FROM public.ecr.aws/lambda/python:3.8

# enable easier debugging
ENV PYTHONBUFFERED 1

# install and setup packages, including poetry
RUN pip install --upgrade pip && pip install curl && \
    curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | \
    POETRY_HOME=/opt/poetry python && \
    cd /usr/local/bin && \
    ln -s /opt/poetry/bin/poetry && \
    poetry config virtualenvs.create false

# copy poetry.lock* in case it doesn't exist in the repo
COPY ./pyproject.toml ./poetry.lock* __init__.py ./

# install project dependencies
RUN poetry install --no-root --no-dev

# add dependencies for training
#ARG TRAINING=false
#RUN bash -c "if [ $INSTALL_JUPYTER == 'true' ] ; then pip install jupyter ; fi"

# copy project
COPY /src/ ./src/

CMD [ "/var/task/src/lambda_function.lambda_handler" ]