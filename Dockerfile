FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /workspace

COPY requirements.txt requirements.lock.txt pyproject.toml ./
COPY src ./src
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.lock.txt \
    && python -m pip install --no-cache-dir --no-deps -e .

COPY . .

CMD ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider"]
