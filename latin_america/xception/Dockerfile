FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=America/Sao_Paulo

# Dependências de sistema para cartopy/rasterio + cron
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cron \
    proj-bin libproj-dev \
    libgeos-dev \
    gdal-bin libgdal-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# App dirs
WORKDIR /app
RUN mkdir -p /var/log

# Requisitos
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Source code
COPY operational_forecast.py /app/operational_forecast.py
COPY xception.py /app/xception.py
COPY latin_america_xception_gsmap_mvk.pth /app/latin_america_xception_gsmap_mvk.pth
COPY eo_base_2020_clean_3600x1800.png /app/eo_base_2020_clean_3600x1800.png

# Crontab setup
COPY crontab /etc/cron.d/nowcasting-cron
RUN chmod 0644 /etc/cron.d/nowcasting-cron && crontab /etc/cron.d/nowcasting-cron

# Keep container running and execute cron
CMD printenv > /etc/environment && cron -f

