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
WORKDIR /tests
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

COPY --from=tdd-build /src/build/tddn_engine /usr/local/bin/tddn_engine
COPY tdd_runtime/LICENSE.TDDLinux /usr/share/licenses/tddn/LICENSE.TDDLinux
COPY solver_worker.py /usr/local/libexec/tddn_solver_worker.py
COPY tdd_backend.py evaluate.py grade.py test.sh /tests/
COPY trusted/ /tests/trusted/
COPY heldout/ /tests/heldout/

# The verifier runs as root. Submitted planners run as uid/gid 65534 and
# cannot traverse /tests or read the sealed manifest.
RUN chmod 700 /tests /tests/heldout /tests/trusted \
 && chmod 600 /tests/heldout/cases.json /tests/heldout/MANIFEST.json \
 && chmod 755 /usr/local/libexec/tddn_solver_worker.py /tests/test.sh

CMD ["bash", "/tests/test.sh"]

# Sealed scoring band: read at import, 0400 root:root so the demoted solver child
# (uid 65534) cannot read it.
COPY anchors.json /tests/anchors.json
RUN chown root:root /tests/anchors.json && chmod 0400 /tests/anchors.json
