# First stage: Builder
FROM python:3.11 as builder

WORKDIR /app

# Install Poetry
RUN pip install poetry
# Copy only the necessary files for installing dependencies to cache them in Docker layer
COPY pyproject.toml poetry.lock* /app/

# Disable virtualenvs created by Poetry
# Install dependencies in the system
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-dev


# Second stage: Application
FROM python:3.11
WORKDIR /app

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    nginx \
    apt-utils \
    build-essential \
    net-tools \
    curl \
    wget \
 && rm -rf /var/lib/apt/lists/*

# Install gunicorn globally in the second stage
RUN pip install gunicorn

# Copy installed dependencies from the builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
# Copy application code and scripts
COPY app /opt/program
WORKDIR /opt/program

# Expose the port that NGINX will run on
EXPOSE 80

# Make sure the 'serve' script is executable
RUN chmod +x /opt/program/serve

# Set the serve script as the entrypoint
ENTRYPOINT ["/opt/program/serve"]