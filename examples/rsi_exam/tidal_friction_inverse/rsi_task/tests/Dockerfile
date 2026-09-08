FROM python:3.11-slim

RUN pip install --no-cache-dir numpy==2.1.3 scipy==1.14.1
RUN useradd -r -s /bin/false solver
WORKDIR /tests
COPY problem.py prepare_heldout.py grade.py test.sh anchors.json /tests/
RUN python /tests/prepare_heldout.py
RUN chmod +x /tests/test.sh
