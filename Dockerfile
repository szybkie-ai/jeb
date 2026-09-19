FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY jeb ./jeb
RUN pip install --no-cache-dir uv && uv pip install --system --no-cache .
EXPOSE 8020
ENTRYPOINT ["jeb"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8020"]
