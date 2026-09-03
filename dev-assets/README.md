# dev-assets

This directory contains various assets used in the development of the project.

## How to develop in an isolated environment

Refer to the following instructions to set up an isolated development environment:

### Prerequisites

Install [Docker](https://www.docker.com/get-started) on your machine.

### Process

Refer to the steps below to set up and access the isolated development environment:

1. Open a terminal and navigate to the root directory of the project.
1. Change the working directory to `dev-assets`:

   ```bash
   cd dev-assets
   ```

1. Run the following command to create the development environment container:

   ```bash
   docker compose up -d
   ```

1. Run the following command to access the container's shell:

   ```bash
   docker container exec -it avisynthplus-dev bash --login
   ```

1. If you like to use a local Ubuntu archive mirror, run the following command:

   ```bash
   sed -i --regexp-extended 's@archive\.@CCTLD\.archive\.@' /etc/apt/sources.list.d/ubuntu.sources
   ```

   Replace `CCTLD` with your country's top-level domain (e.g., `us`, `de`, `fr`, etc.).
1. Run the following command to refresh the package index inside the container:

   ```bash
   apt update
   ```

1. Run the following command to install the dependencies required for building the build dependencies package:

   ```bash
   apt install -y equivs
   ```

1. Run the following commands to build and install the build dependencies metapackage:

   ```bash
   cd /project/dev-assets
   equivs-build dev-assets/avisynthplus.ctl
   apt install ./avisynthplus-build-deps_1.0_all.deb
   ```

   You can now proceed with the regular building instructions in [the main project README](../README.md), the project files are mounted at `/project`.
