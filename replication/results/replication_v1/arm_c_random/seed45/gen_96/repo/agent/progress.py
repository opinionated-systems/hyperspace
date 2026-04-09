"""
Progress tracking utilities for long-running operations.

Provides progress bars, status updates, and timing information
for batch processing and other multi-step operations.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class ProgressStats:
    """Statistics for a progress tracking session."""
    total: int = 0
    completed: int = 0
    failed: int = 0
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    
    @property
    def elapsed(self) -> float:
        """Elapsed time in seconds."""
        end = self.end_time or time.time()
        return end - self.start_time
    
    @property
    def remaining(self) -> int:
        """Number of items remaining."""
        return self.total - self.completed - self.failed
    
    @property
    def percent_complete(self) -> float:
        """Percentage complete (0-100)."""
        if self.total == 0:
            return 0.0
        return (self.completed / self.total) * 100
    
    @property
    def items_per_second(self) -> float:
        """Processing rate."""
        elapsed = self.elapsed
        if elapsed == 0:
            return 0.0
        return self.completed / elapsed
    
    @property
    def estimated_remaining_time(self) -> float:
        """Estimated seconds remaining."""
        rate = self.items_per_second
        if rate == 0:
            return 0.0
        return self.remaining / rate
    
    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "total": self.total,
            "completed": self.completed,
            "failed": self.failed,
            "remaining": self.remaining,
            "percent_complete": round(self.percent_complete, 1),
            "elapsed_seconds": round(self.elapsed, 2),
            "items_per_second": round(self.items_per_second, 2),
            "estimated_remaining_seconds": round(self.estimated_remaining_time, 2),
        }


class ProgressTracker:
    """Track progress of a multi-step operation.
    
    Example:
        tracker = ProgressTracker(total=100, callback=print)
        for i in range(100):
            # Do work
            tracker.update(completed=i+1)
        tracker.finish()
    """
    
    def __init__(
        self,
        total: int,
        callback: Optional[Callable[[str], None]] = None,
        update_interval: float = 1.0,
    ):
        """Initialize progress tracker.
        
        Args:
            total: Total number of items to process
            callback: Function to call with progress updates
            update_interval: Minimum seconds between callback invocations
        """
        self.stats = ProgressStats(total=total)
        self.callback = callback
        self.update_interval = update_interval
        self._last_update = 0.0
    
    def update(
        self,
        completed: Optional[int] = None,
        failed: Optional[int] = None,
        message: Optional[str] = None,
    ) -> None:
        """Update progress.
        
        Args:
            completed: New completed count (or increment if None)
            failed: New failed count (or increment if None)
            message: Optional status message
        """
        if completed is not None:
            self.stats.completed = completed
        if failed is not None:
            self.stats.failed = failed
        
        # Check if we should invoke callback
        now = time.time()
        if self.callback and (now - self._last_update >= self.update_interval):
            self._notify(message)
            self._last_update = now
    
    def increment(self, success: bool = True, message: Optional[str] = None) -> None:
        """Increment progress by one item.
        
        Args:
            success: Whether the item was processed successfully
            message: Optional status message
        """
        if success:
            self.stats.completed += 1
        else:
            self.stats.failed += 1
        
        now = time.time()
        if self.callback and (now - self._last_update >= self.update_interval):
            self._notify(message)
            self._last_update = now
    
    def _notify(self, message: Optional[str] = None) -> None:
        """Invoke callback with current status."""
        if not self.callback:
            return
        
        stats = self.stats
        percent = stats.percent_complete
        elapsed = stats.elapsed
        rate = stats.items_per_second
        eta = stats.estimated_remaining_time
        
        # Format time nicely
        elapsed_str = self._format_time(elapsed)
        eta_str = self._format_time(eta) if eta > 0 else "?"
        
        status = (
            f"[{percent:5.1f}%] "
            f"{stats.completed}/{stats.total} "
            f"({stats.failed} failed) "
            f"| {rate:.1f} items/s "
            f"| Elapsed: {elapsed_str} "
            f"| ETA: {eta_str}"
        )
        
        if message:
            status += f" | {message}"
        
        self.callback(status)
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds as human-readable time."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m{secs:02d}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h{minutes:02d}m"
    
    def finish(self, message: Optional[str] = None) -> dict:
        """Mark operation as complete.
        
        Args:
            message: Optional final status message
            
        Returns:
            Final statistics dictionary
        """
        self.stats.end_time = time.time()
        
        if self.callback:
            final_msg = message or "Complete"
            self._notify(final_msg)
        
        return self.stats.to_dict()


def track_progress(
    items: list,
    process_fn: Callable,
    callback: Optional[Callable[[str], None]] = None,
    update_interval: float = 1.0,
) -> list:
    """Process a list of items with progress tracking.
    
    Args:
        items: List of items to process
        process_fn: Function to process each item
        callback: Progress callback function
        update_interval: Minimum seconds between updates
        
    Returns:
        List of results from process_fn
    """
    tracker = ProgressTracker(
        total=len(items),
        callback=callback,
        update_interval=update_interval,
    )
    
    results = []
    for item in items:
        try:
            result = process_fn(item)
            results.append(result)
            tracker.increment(success=True)
        except Exception as e:
            results.append(e)
            tracker.increment(success=False, message=str(e))
    
    tracker.finish()
    return results
