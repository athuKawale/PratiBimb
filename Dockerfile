FROM nvidia/cuda:12.4.0-base-ubuntu24.04

# Install essential system packages
RUN apt-get update && \
    apt-get install -y wget git nano build-essential && \
    apt-get clean

# Copy scripts and application into the image
COPY environment/anaconda_setup.sh /anaconda_setup.sh
COPY environment/build_linux.sh /build_linux.sh
COPY main.py /main.py

# Ensure scripts are executable
RUN chmod +x /anaconda_setup.sh /build_linux.sh

# Install Anaconda
RUN bash /anaconda_setup.sh

# Add Anaconda to PATH for all subsequent operations
ENV PATH="/root/anaconda3/bin:${PATH}"

# Create Conda environment with Python 3.11
RUN conda create -n pratibimb python=3.11 -y

# Build dependencies in the correct conda environment
RUN /bin/bash -c "source activate pratibimb && bash /build_linux.sh"

# Set default command to activate conda environment and launch API
CMD ["/bin/bash", "-c", "source activate pratibimb && python main.py"]
