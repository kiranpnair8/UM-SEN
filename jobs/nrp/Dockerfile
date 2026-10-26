FROM nvidia/cuda:12.1.1-devel-ubuntu22.04@sha256:327c9e046fbf662275be0934742f7e5412f9b24402ee90bf4d649c1a21707912
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3.10-dev git ca-certificates build-essential \
    libglib2.0-0 libgl1 util-linux && rm -rf /var/lib/apt/lists/*
RUN python3.10 -m pip install --no-cache-dir \
    torch==2.5.1+cu121 torchvision==0.20.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
COPY requirements.txt /opt/nrp/requirements.txt
RUN python3.10 -m pip install --no-cache-dir -r /opt/nrp/requirements.txt && \
    python3.10 -m pip check && python3.10 -m pip freeze > /opt/nrp/pip-freeze.txt
RUN useradd --uid 1000 --create-home --shell /bin/bash nrp
COPY nrp_runner.py entrypoint.sh git-askpass.sh /opt/nrp/
RUN sed -i 's/\r$//' /opt/nrp/*.sh && chmod 755 /opt/nrp/*.sh
ENV HOME=/home/nrp PATH=/usr/local/bin:/usr/bin:/bin GIT_ASKPASS=/opt/nrp/git-askpass.sh GIT_TERMINAL_PROMPT=0
USER 1000:1000
ENTRYPOINT ["/opt/nrp/entrypoint.sh"]
