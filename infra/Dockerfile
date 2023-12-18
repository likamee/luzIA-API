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
COPY ./pyproject.toml ./
COPY ./poetry.lock ./
COPY ./app/ ./app
COPY ./app/boot.sh ./app/boot.sh
COPY ./.env ./app/.env

# Create .env file
RUN printenv | grep -v "no_proxy" >> ./app/.env

RUN poetry install

RUN chmod +x ./app/boot.sh

RUN chown -R $USER:$USER /home/$USER
USER $USER

WORKDIR /home/$USER/app

ENTRYPOINT ["sh", "./boot.sh"]
