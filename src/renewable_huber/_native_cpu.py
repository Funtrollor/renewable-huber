"""Import bridge for the separately built native CPU extension."""

from _renewable_huber_native_cpu import (
    NativeCpuEngine,
    version,
)

__all__ = ["NativeCpuEngine", "version"]
