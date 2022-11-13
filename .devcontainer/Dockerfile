ARG VARIANT="11-bullseye"
FROM mcr.microsoft.com/vscode/devcontainers/java-${VARIANT}
RUN wget -q0- https://get.nextflow.io | bash && mv nextflow /usr/local/bin/
COPY library-scripts/github-debian.sh /tmp/library-scripts/
RUN apt-get update && bash /tmp/library-scripts/github-debian.sh