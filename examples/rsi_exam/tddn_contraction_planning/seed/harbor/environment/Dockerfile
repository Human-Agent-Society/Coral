FROM ubuntu:22.04 AS tdd-build

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      build-essential cmake libxtensor-dev nlohmann-json3-dev ninja-build xtl-dev \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY tdd_runtime/ /src/
RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
 && cmake --build build -j2

FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      ripgrep \
 && rm -rf /var/lib/apt/lists/*
COPY --from=tdd-build /src/build/tddn_engine /usr/local/bin/tddn_engine
COPY tdd_runtime/LICENSE.TDDLinux /usr/share/licenses/tddn/LICENSE.TDDLinux
COPY methods/ /app/methods/
COPY data/ /app/data/
COPY tdd_backend.py evaluate_visible.py selfcheck.py /app/

CMD ["bash"]
