FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md LICENSE requirements.txt ./
COPY src ./src
COPY tests ./tests

RUN python -m pip install --no-cache-dir -e ".[test]"

CMD ["python", "-m", "genesis_fintech.check"]
