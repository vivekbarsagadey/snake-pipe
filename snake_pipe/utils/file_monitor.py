"""Low-level file monitoring utilities for the watcher system.

This module provides cross-platform file monitoring utilities and
system-level integration for the file watcher service.
"""

import asyncio
import logging
import os
import platform
import psutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Tuple
from enum import Enum, auto

logger = logging.getLogger(__name__)


class MonitoringBackend(Enum):
    """Available file monitoring backends."""
    WATCHDOG = "watchdog"          # Cross-platform watchdog library
    INOTIFY = "inotify"           # Linux native inotify
    KQUEUE = "kqueue"             # BSD/macOS native kqueue
    POLLING = "polling"           # Fallback polling mechanism


@dataclass
class SystemLimits:
    """System limits for file monitoring."""
    max_watches: int
    max_open_files: int
    available_memory: int
    cpu_count: int
    platform: str


@dataclass
class MonitoringCapabilities:
    """Capabilities of the current monitoring system."""
    backend: MonitoringBackend
    native_recursive: bool
    event_types: List[str]
    max_watches: int
    performance_rating: int  # 1-10 scale
    
    
class FileMonitor:
    """Low-level file monitoring utilities and system integration."""
    
    def __init__(self):
        self.system_limits = self._detect_system_limits()
        self.capabilities = self._detect_capabilities()
        self.active_watches: Set[str] = set()
        self.performance_stats = {
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'watch_count': 0,
            'event_rate': 0.0
        }
    
    def _detect_system_limits(self) -> SystemLimits:
        """Detect system limits for file monitoring."""
        system = platform.system().lower()
        
        # Get memory info
        memory = psutil.virtual_memory()
        available_memory = memory.available
        
        # Get CPU count
        cpu_count = psutil.cpu_count()
        
        # Get file descriptor limits
        try:
            import resource
            max_open_files = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
        except (ImportError, OSError):
            max_open_files = 1024  # Conservative default
        
        # Platform-specific watch limits
        max_watches = self._get_max_watches(system)
        
        return SystemLimits(
            max_watches=max_watches,
            max_open_files=max_open_files,
            available_memory=available_memory,
            cpu_count=cpu_count,
            platform=system
        )
    
    def _get_max_watches(self, system: str) -> int:
        """Get maximum number of watches for the system."""
        if system == 'linux':
            try:
                # Read inotify max_user_watches
                with open('/proc/sys/fs/inotify/max_user_watches', 'r') as f:
                    return int(f.read().strip())
            except (OSError, ValueError):
                return 8192  # Default Linux limit
        
        elif system in ['darwin', 'freebsd']:
            # macOS/BSD typically have lower limits
            return 1024
        
        elif system == 'windows':
            # Windows has different limits
            return 2048
        
        else:
            return 1024  # Conservative default
    
    def _detect_capabilities(self) -> MonitoringCapabilities:
        """Detect monitoring capabilities of the current system."""
        system = self.system_limits.platform
        
        if system == 'linux':
            # Linux has excellent inotify support
            backend = MonitoringBackend.INOTIFY
            native_recursive = True
            event_types = ['IN_CREATE', 'IN_MODIFY', 'IN_DELETE', 'IN_MOVE']
            performance_rating = 9
            
        elif system == 'darwin':
            # macOS has kqueue support
            backend = MonitoringBackend.KQUEUE
            native_recursive = False  # kqueue doesn't support recursive natively
            event_types = ['NOTE_WRITE', 'NOTE_DELETE', 'NOTE_RENAME']
            performance_rating = 7
            
        elif system == 'freebsd':
            # FreeBSD kqueue support
            backend = MonitoringBackend.KQUEUE
            native_recursive = False
            event_types = ['NOTE_WRITE', 'NOTE_DELETE', 'NOTE_RENAME']
            performance_rating = 7
            
        elif system == 'windows':
            # Windows ReadDirectoryChangesW via watchdog
            backend = MonitoringBackend.WATCHDOG
            native_recursive = True
            event_types = ['FILE_ACTION_ADDED', 'FILE_ACTION_MODIFIED', 'FILE_ACTION_REMOVED']
            performance_rating = 6
            
        else:
            # Fallback to polling
            backend = MonitoringBackend.POLLING
            native_recursive = True
            event_types = ['created', 'modified', 'deleted']
            performance_rating = 3
        
        return MonitoringCapabilities(
            backend=backend,
            native_recursive=native_recursive,
            event_types=event_types,
            max_watches=self.system_limits.max_watches,
            performance_rating=performance_rating
        )
    
    def can_monitor_path(self, path: Path, recursive: bool = True) -> Tuple[bool, str]:
        """Check if a path can be monitored with current system capabilities."""
        if not path.exists():
            return False, f"Path does not exist: {path}"
        
        if not path.is_dir():
            return False, f"Path is not a directory: {path}"
        
        # Check permissions
        if not os.access(str(path), os.R_OK):
            return False, f"No read permission for path: {path}"
        
        # Estimate watch count needed
        watch_count = self._estimate_watch_count(path, recursive)
        
        # Check against system limits
        current_watches = len(self.active_watches)
        if current_watches + watch_count > self.capabilities.max_watches * 0.8:
            return False, f"Would exceed watch limit: {current_watches + watch_count} > {self.capabilities.max_watches}"
        
        # Check memory requirements
        estimated_memory = watch_count * 1024  # Rough estimate: 1KB per watch
        if estimated_memory > self.system_limits.available_memory * 0.1:
            return False, f"Would use too much memory: {estimated_memory} bytes"
        
        return True, "Path can be monitored"
    
    def _estimate_watch_count(self, path: Path, recursive: bool) -> int:
        """Estimate number of watches needed for a path."""
        if not recursive:
            return 1
        
        try:
            # Quick directory count estimation
            count = 1  # Root directory
            
            # Sample first few levels to estimate total
            for i, (root, dirs, files) in enumerate(os.walk(str(path))):
                if i > 100:  # Limit sampling to avoid long delays
                    # Estimate based on sample
                    avg_dirs_per_level = count / max(1, i)
                    estimated_total = count * (avg_dirs_per_level ** 3)  # Rough exponential estimate
                    return min(int(estimated_total), 50000)  # Cap at reasonable limit
                
                count += len(dirs)
            
            return count
            
        except (OSError, PermissionError):
            return 1000  # Conservative estimate when we can't scan
    
    def optimize_for_workload(self, expected_files: int, expected_changes_per_sec: int) -> Dict[str, Any]:
        """Get optimization recommendations for expected workload."""
        recommendations = {
            'backend': self.capabilities.backend.value,
            'batch_size': 100,
            'debounce_period': 0.5,
            'memory_limit': '100MB',
            'warnings': []
        }
        
        # Adjust batch size based on change rate
        if expected_changes_per_sec > 100:
            recommendations['batch_size'] = 200
            recommendations['debounce_period'] = 0.2
            recommendations['warnings'].append('High change rate detected')
        
        # Adjust for large file counts
        if expected_files > 100000:
            recommendations['memory_limit'] = '500MB'
            recommendations['warnings'].append('Large file count requires more memory')
        
        # Platform-specific optimizations
        if self.system_limits.platform == 'linux':
            recommendations['use_inotify'] = True
        elif self.system_limits.platform == 'darwin':
            recommendations['batch_size'] = 50  # macOS kqueue is less efficient
            recommendations['warnings'].append('macOS kqueue has performance limitations')
        
        return recommendations
    
    async def monitor_performance(self, duration: float = 60.0) -> Dict[str, float]:
        """Monitor system performance during file watching."""
        start_time = time.time()
        samples = []
        
        while time.time() - start_time < duration:
            try:
                # Get current process stats
                process = psutil.Process()
                cpu_percent = process.cpu_percent()
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                
                samples.append({
                    'timestamp': time.time(),
                    'cpu_percent': cpu_percent,
                    'memory_mb': memory_mb,
                    'watch_count': len(self.active_watches)
                })
                
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Error monitoring performance: {e}")
                break
        
        if not samples:
            return {}
        
        # Calculate averages
        avg_cpu = sum(s['cpu_percent'] for s in samples) / len(samples)
        avg_memory = sum(s['memory_mb'] for s in samples) / len(samples)
        max_memory = max(s['memory_mb'] for s in samples)
        
        return {
            'duration': time.time() - start_time,
            'avg_cpu_percent': avg_cpu,
            'avg_memory_mb': avg_memory,
            'max_memory_mb': max_memory,
            'sample_count': len(samples),
            'watch_count': len(self.active_watches)
        }
    
    def add_watch(self, path: str) -> bool:
        """Register a new watch path."""
        if len(self.active_watches) >= self.capabilities.max_watches * 0.9:
            logger.warning(f"Approaching watch limit: {len(self.active_watches)}")
            return False
        
        self.active_watches.add(path)
        self.performance_stats['watch_count'] = len(self.active_watches)
        return True
    
    def remove_watch(self, path: str) -> bool:
        """Unregister a watch path."""
        if path in self.active_watches:
            self.active_watches.remove(path)
            self.performance_stats['watch_count'] = len(self.active_watches)
            return True
        return False
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information for monitoring."""
        return {
            'platform': self.system_limits.platform,
            'backend': self.capabilities.backend.value,
            'max_watches': self.capabilities.max_watches,
            'max_open_files': self.system_limits.max_open_files,
            'available_memory_mb': self.system_limits.available_memory / 1024 / 1024,
            'cpu_count': self.system_limits.cpu_count,
            'performance_rating': self.capabilities.performance_rating,
            'native_recursive': self.capabilities.native_recursive,
            'active_watches': len(self.active_watches),
            'supported_events': self.capabilities.event_types
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check of monitoring system."""
        health = {
            'status': 'healthy',
            'issues': [],
            'performance': self.performance_stats.copy()
        }
        
        # Check watch count
        watch_ratio = len(self.active_watches) / self.capabilities.max_watches
        if watch_ratio > 0.8:
            health['issues'].append(f'High watch count: {watch_ratio:.1%} of limit')
            if watch_ratio > 0.95:
                health['status'] = 'critical'
        
        # Check memory usage
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            if memory_mb > 1000:  # 1GB
                health['issues'].append(f'High memory usage: {memory_mb:.1f}MB')
        except Exception:
            pass
        
        # Check CPU usage
        if self.performance_stats.get('cpu_usage', 0) > 5.0:
            health['issues'].append(f"High CPU usage: {self.performance_stats['cpu_usage']:.1f}%")
        
        if health['issues'] and health['status'] == 'healthy':
            health['status'] = 'warning'
        
        return health


# Global file monitor instance
_file_monitor = None


def get_file_monitor() -> FileMonitor:
    """Get global file monitor instance."""
    global _file_monitor
    if _file_monitor is None:
        _file_monitor = FileMonitor()
    return _file_monitor


async def check_system_capabilities() -> Dict[str, Any]:
    """Check system capabilities for file monitoring."""
    monitor = get_file_monitor()
    return monitor.get_system_info()


async def optimize_for_path(path: Path, recursive: bool = True) -> Dict[str, Any]:
    """Get optimization recommendations for monitoring a specific path."""
    monitor = get_file_monitor()
    
    can_monitor, reason = monitor.can_monitor_path(path, recursive)
    if not can_monitor:
        return {
            'can_monitor': False,
            'reason': reason,
            'recommendations': []
        }
    
    # Estimate workload
    watch_count = monitor._estimate_watch_count(path, recursive)
    
    # Get optimization recommendations
    recommendations = monitor.optimize_for_workload(
        expected_files=watch_count * 10,  # Estimate files per directory
        expected_changes_per_sec=10  # Conservative estimate
    )
    
    return {
        'can_monitor': True,
        'estimated_watches': watch_count,
        'recommendations': recommendations,
        'system_info': monitor.get_system_info()
    }
