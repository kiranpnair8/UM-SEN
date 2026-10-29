# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04@sha256:327c9e046fbf662275be0934742f7e5412f9b24402ee90bf4d649c1a21707912
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
RUN apt-get update && apt-get install -y --no-install-recommends python3.10 python3-pip python3.10-dev git ca-certificates build-essential libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*
RUN python3.10 -m pip install --no-cache-dir torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
COPY requirements.txt /opt/simple/requirements.txt
RUN python3.10 -m pip install --no-cache-dir -r /opt/simple/requirements.txt && python3.10 -m pip check && python3.10 -m pip freeze > /opt/simple/pip-freeze.txt
COPY git-askpass.sh /opt/simple/git-askpass.sh
RUN sed -i 's/\r$//' /opt/simple/git-askpass.sh && chmod 755 /opt/simple/git-askpass.sh
# Optional BuildKit secret is used only during clone, never stored in Git/ENV.
RUN --mount=type=secret,id=github_token \
    export NRP_GIT_TOKEN="$(if [ -f /run/secrets/github_token ]; then cat /run/secrets/github_token; fi)" GIT_ASKPASS=/opt/simple/git-askpass.sh GIT_TERMINAL_PROMPT=0; \
    git clone --filter=blob:none https://github.com/kiranpnair8/UM-SEN.git /opt/scientific && \
    git -C /opt/scientific checkout --detach 01ad13da81eed92b64247c5dc3fcd44608f31a1f && \
    git -C /opt/scientific show 7cd92a8d7402957083c855823dca906d5ac681b6:configs/gradient_flow/controller_followup.yml > /dev/null && \
    test "$(git -C /opt/scientific rev-parse HEAD)" = 01ad13da81eed92b64247c5dc3fcd44608f31a1f && \
    test -z "$(git -C /opt/scientific status --porcelain)"
COPY runner.py entrypoint.sh /opt/simple/
RUN sed -i 's/\r$//' /opt/simple/entrypoint.sh && chmod 755 /opt/simple/entrypoint.sh && useradd --uid 1000 --create-home nrp
ENV HOME=/home/nrp PATH=/usr/local/bin:/usr/bin:/bin
USER 1000:1000
ENTRYPOINT ["/opt/simple/entrypoint.sh"]
