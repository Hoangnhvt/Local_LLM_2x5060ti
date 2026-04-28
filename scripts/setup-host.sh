#!/usr/bin/env bash
# One-time host setup for Kurt's Brain on Ubuntu Pro 22.04 / 24.04.
# Run as root on 192.168.3.5.
#
#   sudo bash scripts/setup-host.sh
#
# Idempotent — safe to re-run.
set -euo pipefail

if [[ $EUID -ne 0 ]]; then echo "run as root"; exit 1; fi

UBUNTU_CODENAME=$(. /etc/os-release && echo "$VERSION_CODENAME")
DRIVER_BRANCH=570   # Blackwell needs >= 565

echo "==> apt baseline"
apt-get update
apt-get install -y curl ca-certificates gnupg lsb-release software-properties-common \
                   build-essential git jq htop nvtop tmux

echo "==> NVIDIA driver $DRIVER_BRANCH"
add-apt-repository -y ppa:graphics-drivers/ppa || true
apt-get update
apt-get install -y "nvidia-driver-$DRIVER_BRANCH"

echo "==> Docker CE"
if ! command -v docker >/dev/null; then
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
    gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  chmod a+r /etc/apt/keyrings/docker.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
    https://download.docker.com/linux/ubuntu $UBUNTU_CODENAME stable" \
    > /etc/apt/sources.list.d/docker.list
  apt-get update
  apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
fi
usermod -aG docker kurt || true

echo "==> NVIDIA Container Toolkit"
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  > /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update
apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

echo "==> models cache dir"
install -d -o kurt -g kurt -m 0775 /srv/models

echo "==> firewall (open LiteLLM port on the OpenVPN subnet only)"
if command -v ufw >/dev/null && ufw status | grep -q active; then
  ufw allow from 10.0.0.0/8 to any port 4000 proto tcp || true
fi

echo
echo "==> verify"
nvidia-smi || { echo "driver not loaded — REBOOT, then re-run"; exit 1; }
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi

echo
echo "DONE. If nvidia-smi failed, reboot and re-run."
