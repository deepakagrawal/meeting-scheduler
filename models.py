"""Data models for the meeting assignment solver."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class OptimizationMode(Enum):
    """Optimization strategy."""
    FOCUS_TIME = "focus_time"       # Maximize contiguous free blocks (calendar defrag)
    MIN_DISRUPTION = "min_disruption"  # Minimize scheduling disruption


@dataclass
class MeetingRequest:
    """A meeting that needs to be scheduled or rescheduled.

    All times are expressed in discrete slots (e.g., 15-minute intervals).
    """
    duration_slots: int
    available_slots: list[int]          # Candidate start slots
    attendees: list[int]                # 0-based attendee indices
    current_start_slot: Optional[int] = None  # None = new meeting
    is_blocked_time: bool = False       # True = personal focus block, not a real meeting
    preferred_slots: list[int] = field(default_factory=list)  # Subset of available_slots preferred (e.g., in-person)
    move_penalty: float = 1.0           # Cost weight for moving this meeting
    min_duration_slots: Optional[int] = None  # For dynamic-duration meetings
    occurrence_day_offset: Optional[int] = None  # For recurring meeting linkage

    @property
    def is_dynamic_duration(self) -> bool:
        return (
            self.min_duration_slots is not None
            and self.min_duration_slots != self.duration_slots
        )

    @property
    def effective_available_slots(self) -> list[int]:
        """Available slots including the current slot if set."""
        slots = set(self.available_slots)
        if self.current_start_slot is not None:
            slots.add(self.current_start_slot)
        return sorted(slots)


@dataclass
class AttendeeContext:
    """Calendar metadata for a single attendee."""
    work_hours: list[int] = field(default_factory=list)     # Slots within work hours
    meeting_hours: list[int] = field(default_factory=list)  # Slots within meeting hours
    busy_slots: list[int] = field(default_factory=list)     # Hard commitments
    tentative_slots: list[int] = field(default_factory=list)


@dataclass
class FocusTimeConstraint:
    """Guarantee minimum focus time for an attendee."""
    attendee_index: int
    min_focus_time: float  # Minimum guaranteed focus time (in slots)


@dataclass
class ScheduleRequest:
    """Top-level solver input."""
    total_slots: int                          # Time horizon in slots
    slot_duration_minutes: int = 15           # Duration of each slot
    min_focus_block_slots: int = 4            # Min contiguous free slots to count as focus time (e.g., 1 hour)
    meetings: list[MeetingRequest] = field(default_factory=list)
    attendee_free_slots: list[list[int]] = field(default_factory=list)  # Per-attendee potential focus slots
    attendee_contexts: list[AttendeeContext] = field(default_factory=list)
    focus_time_constraints: list[FocusTimeConstraint] = field(default_factory=list)
    optimization_mode: OptimizationMode = OptimizationMode.FOCUS_TIME
    timeout_seconds: int = 30
    objective_weights: dict[str, float] = field(default_factory=dict)
    focus_time_cap: Optional[float] = None   # Cap focus time per attendee (prevents over-optimization)
    simplified: bool = False                 # Faster but less accurate


@dataclass
class MeetingResult:
    """Solver output for a single meeting."""
    start_slot: int
    duration_slots: int


@dataclass
class ScheduleResponse:
    """Solver output."""
    changed_meetings: dict[int, MeetingResult]  # meeting_index -> result
    objective_values: dict[str, float] = field(default_factory=dict)
    status: str = "OPTIMAL"
