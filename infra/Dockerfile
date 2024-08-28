# Base image
FROM python:3.10-slim-bullseye

# Set environment variables
ENV POETRY_VIRTUALENVS_IN_PROJECT=true \
  POETRY_HOME="/home/poetry" \
  VENV_PATH="/home/luzia-api/.venv" \
  USER="luzia-api" \
  PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"

# Update system packages, install dependencies, and create user
RUN useradd $USER && \
  apt-get update && \
  apt-get install -y git build-essential locales ffmpeg curl && \
  pip3 install --upgrade pip && \
  pip3 install poetry

# Set working directory
WORKDIR /home/$USER

# Configure locale
RUN sed -i '/pt_BR.UTF-8/s/^# //g' /etc/locale.gen && \
  locale-gen && \
  locale

# Set locale environment variables
ENV LANG pt_BR.UTF-8 \
  LANGUAGE pt_BR:en \
  LC_ALL pt_BR.UTF-8

# COPY EVERYTING FROM THE CURRENT DIRECTORY TO THE WORKDIR
COPY ./pyproject.toml ./poetry.lock ./app/ ./app/boot.sh ./.env /home/$USER/app/
COPY ./credentials.json /home/$USER/app/credentials/

WORKDIR /home/$USER/app

RUN poetry install

RUN chmod +x ./boot.sh

RUN chown -R $USER:$USER /home/$USER
USER $USER



ENTRYPOINT ["sh", "./boot.sh"]
