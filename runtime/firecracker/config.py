"""
Firecracker MicroVM Configuration

Configuration dataclasses for Firecracker VM settings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class VMState(StrEnum):
    """State of a MicroVM."""

    CREATED = "created"  # VM created but not started
    STARTING = "starting"  # VM is booting
    RUNNING = "running"  # VM is running and ready
    PAUSED = "paused"  # VM is paused
    STOPPING = "stopping"  # VM is shutting down
    STOPPED = "stopped"  # VM has stopped
    ERROR = "error"  # VM encountered an error


@dataclass
class FirecrackerConfig:
    """
    Configuration for Firecracker MicroVM.

    Environment variables can override defaults:
    - FIRECRACKER_VCPU_COUNT: Number of vCPUs
    - FIRECRACKER_MEM_SIZE_MIB: Memory size in MiB
    - FIRECRACKER_KERNEL_PATH: Path to kernel image
    - FIRECRACKER_ROOTFS_PATH: Path to root filesystem
    - FIRECRACKER_TIMEOUT: Execution timeout in seconds
    """

    # VM Resources
    vcpu_count: int = field(default_factory=lambda: int(os.getenv("FIRECRACKER_VCPU_COUNT", "1")))
    mem_size_mib: int = field(
        default_factory=lambda: int(os.getenv("FIRECRACKER_MEM_SIZE_MIB", "128"))
    )
    ht_enabled: bool = False  # Hyperthreading

    # Kernel and rootfs images
    kernel_path: str = field(
        default_factory=lambda: os.getenv("FIRECRACKER_KERNEL_PATH", "/var/lib/firecracker/vmlinux")
    )
    kernel_boot_args: str = "console=ttyS0 reboot=k panic=1 pci=off init=/usr/bin/agent"

    rootfs_path: str = field(
        default_factory=lambda: os.getenv(
            "FIRECRACKER_ROOTFS_PATH", "/var/lib/firecracker/rootfs.ext4"
        )
    )
    rootfs_is_readonly: bool = False

    # Networking
    enable_network: bool = False
    tap_device: str | None = None
    host_dev_name: str | None = None  # TAP device name on host
    guest_mac: str | None = None
    guest_ip: str = "172.16.0.2"
    host_ip: str = "172.16.0.1"
    netmask: str = "255.255.255.0"

    # VSOCK for guest communication
    enable_vsock: bool = True
    vsock_cid: int = 3  # Context ID (must be > 2)
    vsock_uds_path: str | None = None

    # Execution settings
    timeout_seconds: int = field(
        default_factory=lambda: int(os.getenv("FIRECRACKER_TIMEOUT", "30"))
    )
    max_output_bytes: int = 1_000_000
    boot_timeout_seconds: int = 5

    # Firecracker binary and socket
    firecracker_path: str = field(
        default_factory=lambda: os.getenv("FIRECRACKER_BINARY", "/usr/local/bin/firecracker")
    )
    jailer_path: str | None = None  # Optional jailer for additional isolation
    socket_path_template: str = "/tmp/firecracker-{vm_id}.socket"

    # Logging
    log_path: str | None = None
    log_level: str = "Warning"  # Error, Warning, Info, Debug
    metrics_path: str | None = None

    # Security
    seccomp_level: int = 2  # 0=disabled, 1=basic, 2=advanced
    use_jailer: bool = False
    jailer_uid: int = 1000
    jailer_gid: int = 1000
    chroot_base_dir: str = "/srv/jailer"

    # Rate limiting (for drives)
    rate_limiter_bandwidth: int | None = None  # Bytes/s
    rate_limiter_ops: int | None = None  # Operations/s

    def get_socket_path(self, vm_id: str) -> str:
        """Get socket path for a specific VM."""
        return self.socket_path_template.format(vm_id=vm_id)

    def to_machine_config(self) -> dict[str, Any]:
        """Convert to Firecracker machine-config API format."""
        return {
            "vcpu_count": self.vcpu_count,
            "mem_size_mib": self.mem_size_mib,
            "ht_enabled": self.ht_enabled,
        }

    def to_boot_source(self) -> dict[str, Any]:
        """Convert to Firecracker boot-source API format."""
        return {
            "kernel_image_path": self.kernel_path,
            "boot_args": self.kernel_boot_args,
        }

    def to_drive_config(self, drive_id: str = "rootfs") -> dict[str, Any]:
        """Convert to Firecracker drive API format."""
        config: dict[str, Any] = {
            "drive_id": drive_id,
            "path_on_host": self.rootfs_path,
            "is_root_device": True,
            "is_read_only": self.rootfs_is_readonly,
        }

        if self.rate_limiter_bandwidth or self.rate_limiter_ops:
            config["rate_limiter"] = {}
            if self.rate_limiter_bandwidth:
                config["rate_limiter"]["bandwidth"] = {
                    "size": self.rate_limiter_bandwidth,
                    "refill_time": 1000,  # ms
                }
            if self.rate_limiter_ops:
                config["rate_limiter"]["ops"] = {
                    "size": self.rate_limiter_ops,
                    "refill_time": 1000,
                }

        return config

    def to_network_config(self, iface_id: str = "eth0") -> dict[str, Any] | None:
        """Convert to Firecracker network-interfaces API format."""
        if not self.enable_network or not self.tap_device:
            return None

        config = {
            "iface_id": iface_id,
            "host_dev_name": self.tap_device,
        }

        if self.guest_mac:
            config["guest_mac"] = self.guest_mac

        return config

    def to_vsock_config(self) -> dict[str, Any] | None:
        """Convert to Firecracker vsock API format."""
        if not self.enable_vsock:
            return None

        return {
            "vsock_id": "vsock0",
            "guest_cid": self.vsock_cid,
            "uds_path": self.vsock_uds_path or f"/tmp/vsock-{self.vsock_cid}.sock",
        }

    def to_logger_config(self) -> dict[str, Any] | None:
        """Convert to Firecracker logger API format."""
        if not self.log_path:
            return None

        return {
            "log_path": self.log_path,
            "level": self.log_level,
            "show_level": True,
            "show_log_origin": True,
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "vcpu_count": self.vcpu_count,
            "mem_size_mib": self.mem_size_mib,
            "ht_enabled": self.ht_enabled,
            "kernel_path": self.kernel_path,
            "kernel_boot_args": self.kernel_boot_args,
            "rootfs_path": self.rootfs_path,
            "rootfs_is_readonly": self.rootfs_is_readonly,
            "enable_network": self.enable_network,
            "tap_device": self.tap_device,
            "guest_ip": self.guest_ip,
            "host_ip": self.host_ip,
            "enable_vsock": self.enable_vsock,
            "vsock_cid": self.vsock_cid,
            "timeout_seconds": self.timeout_seconds,
            "max_output_bytes": self.max_output_bytes,
            "boot_timeout_seconds": self.boot_timeout_seconds,
            "firecracker_path": self.firecracker_path,
            "use_jailer": self.use_jailer,
            "seccomp_level": self.seccomp_level,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FirecrackerConfig:
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def minimal(cls) -> FirecrackerConfig:
        """Create minimal configuration for quick tests."""
        return cls(
            vcpu_count=1,
            mem_size_mib=64,
            enable_network=False,
            enable_vsock=True,
            timeout_seconds=10,
        )

    @classmethod
    def for_code_execution(cls) -> FirecrackerConfig:
        """Create configuration optimized for code execution."""
        return cls(
            vcpu_count=1,
            mem_size_mib=256,
            enable_network=False,
            enable_vsock=True,
            timeout_seconds=30,
            max_output_bytes=10_000_000,
        )

    @classmethod
    def for_network_tasks(cls) -> FirecrackerConfig:
        """Create configuration with networking enabled."""
        return cls(
            vcpu_count=2,
            mem_size_mib=512,
            enable_network=True,
            enable_vsock=True,
            timeout_seconds=60,
        )


@dataclass
class VMResourceLimits:
    """Resource limits for VM operations."""

    max_memory_mib: int = 1024
    max_vcpus: int = 4
    max_disk_size_gib: int = 10
    max_network_bandwidth_mbps: int = 100
    max_concurrent_vms: int = 10


# Default resource limits
DEFAULT_RESOURCE_LIMITS = VMResourceLimits()
