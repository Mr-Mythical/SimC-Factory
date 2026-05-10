FROM simulationcraftorg/simc:latest

USER root

# Keep the worker image lightweight: no Python, only shell tools plus AWS CLI.
RUN set -eux; \
    if command -v apt-get >/dev/null 2>&1; then \
      apt-get update && apt-get install -y --no-install-recommends bash curl jq zip unzip ca-certificates findutils awscli && rm -rf /var/lib/apt/lists/*; \
    elif command -v dnf >/dev/null 2>&1; then \
      dnf install -y bash curl jq zip unzip findutils ca-certificates awscli && dnf clean all; \
    elif command -v yum >/dev/null 2>&1; then \
      yum install -y bash curl jq zip unzip findutils ca-certificates awscli && yum clean all; \
    elif command -v apk >/dev/null 2>&1; then \
      apk add --no-cache bash curl jq zip unzip findutils ca-certificates aws-cli; \
    else \
      echo "Unsupported base image: no known package manager found" >&2; exit 1; \
    fi

RUN set -eux; \
    if command -v simc >/dev/null 2>&1; then \
      ln -sf "$(command -v simc)" /usr/local/bin/simc; \
    else \
      SIMC_PATH="$(find / -type f -name simc 2>/dev/null | head -n 1)"; \
      test -n "$SIMC_PATH"; \
      ln -sf "$SIMC_PATH" /usr/local/bin/simc; \
    fi

WORKDIR /opt/simc-batch
COPY batch_worker.sh /opt/simc-batch/batch_worker.sh
RUN chmod +x /opt/simc-batch/batch_worker.sh

ENTRYPOINT ["/opt/simc-batch/batch_worker.sh"]
