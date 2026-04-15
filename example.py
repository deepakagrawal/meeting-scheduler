"""
Example: Calendar Defrag — rearrange meetings to maximize focus time.

Scenario: 3 attendees, 8-hour day (32 slots of 15 min each).
Several meetings are already scheduled; the solver tries to consolidate
them to create longer contiguous free blocks.
"""

from models import (
    AttendeeContext,
    FocusTimeConstraint,
    MeetingRequest,
    OptimizationMode,
    ScheduleRequest,
)
from solver import Solver


def slot_to_time(slot: int, start_hour: int = 9) -> str:
    """Convert slot index to human-readable time."""
    total_minutes = start_hour * 60 + slot * 15
    h, m = divmod(total_minutes, 60)
    return f"{h:02d}:{m:02d}"


def print_schedule(request: ScheduleRequest, response, label: str = ""):
    """Print a visual schedule."""
    if label:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")

    print(f"\nStatus: {response.status}")
    if response.objective_values:
        for k, v in response.objective_values.items():
            print(f"  {k}: {v:.1f}")

    num_attendees = len(request.attendee_free_slots)

    # Build per-attendee schedule
    for j in range(num_attendees):
        print(f"\n  Attendee {j}:")
        timeline = ["."] * request.total_slots

        # Mark busy (pre-existing)
        ctx = request.attendee_contexts[j] if j < len(request.attendee_contexts) else None
        if ctx:
            for t in ctx.busy_slots:
                if t < request.total_slots:
                    timeline[t] = "B"

        # Mark meetings
        for i, mtg in enumerate(request.meetings):
            if j not in mtg.attendees:
                continue

            # Determine start slot
            if i in response.changed_meetings:
                start = response.changed_meetings[i].start_slot
                dur = response.changed_meetings[i].duration_slots
                moved = mtg.current_start_slot is not None and start != mtg.current_start_slot
            elif mtg.current_start_slot is not None:
                start = mtg.current_start_slot
                dur = mtg.duration_slots
                moved = False
            else:
                continue

            char = f"{i}" if i < 10 else chr(ord('A') + i - 10)
            for dt in range(dur):
                if start + dt < request.total_slots:
                    timeline[start + dt] = f"{'*' if moved else ''}{char}"[-1]

        # Print timeline
        print(f"    {''.join(f'{slot_to_time(t):>6}' for t in range(0, request.total_slots, 4))}")
        print(f"    {''.join(f'{t:>6}' for t in range(0, request.total_slots, 4))}")
        line = "    "
        for t in range(request.total_slots):
            c = timeline[t]
            if c == ".":
                line += " "
            elif c == "B":
                line += "\033[90m#\033[0m"
            else:
                line += f"\033[94m{c}\033[0m"
        print(line)

    # Summary of changes
    if response.changed_meetings:
        print(f"\n  Changes:")
        for i, result in sorted(response.changed_meetings.items()):
            mtg = request.meetings[i]
            old = f"slot {mtg.current_start_slot} ({slot_to_time(mtg.current_start_slot)})" if mtg.current_start_slot is not None else "NEW"
            new = f"slot {result.start_slot} ({slot_to_time(result.start_slot)})"
            print(f"    Meeting {i}: {old} -> {new} (dur={result.duration_slots} slots)")
    else:
        print("\n  No changes needed.")


def example_defrag():
    """Calendar defrag: rearrange existing meetings to maximize focus time.

    Scenario: Attendee 0 has 5 meetings scattered throughout the day with
    small 30-min gaps between them — no time for deep work. The solver
    consolidates them into a tighter block to open up focus time.

    BEFORE (fragmented):
      A0: |M0|  |M1|    |M2|  |M3|    |M4|    — gaps everywhere
      A1: |M0|         |M2|  |M3|              — also fragmented
      A2:     |M1|     |M2|       |M4|

    AFTER (defragged):
      A0: |M0 M1 M2 M4|                       — all back-to-back, afternoon free
      A1: |M0 M2 M3|                           — consolidated
      A2: |M1 M2 M4|                           — consolidated
    """
    TOTAL_SLOTS = 32  # 8-hour day (9:00 - 17:00)

    work_hours = list(range(TOTAL_SLOTS))
    meeting_hours = list(range(4, 28))  # 10:00 - 16:00

    attendee_contexts = [
        AttendeeContext(work_hours=work_hours, meeting_hours=meeting_hours, busy_slots=[], tentative_slots=[]),
        AttendeeContext(work_hours=work_hours, meeting_hours=meeting_hours, busy_slots=[], tentative_slots=[]),
        AttendeeContext(work_hours=work_hours, meeting_hours=meeting_hours, busy_slots=[], tentative_slots=[]),
    ]

    attendee_free_slots = [work_hours[:], work_hours[:], work_hours[:]]

    # Deliberately fragmented: 30min meetings with 30-45min gaps (too short for focus)
    meetings = [
        # Meeting 0: 30min, attendees 0+1, at 9:00 (slot 0)
        MeetingRequest(
            duration_slots=2, available_slots=list(range(0, 28)),
            attendees=[0, 1], current_start_slot=0,
        ),
        # Meeting 1: 30min, attendees 0+2, at 9:45 (slot 3) — 15min gap
        MeetingRequest(
            duration_slots=2, available_slots=list(range(0, 28)),
            attendees=[0, 2], current_start_slot=3,
        ),
        # Meeting 2: 30min, attendees 0+1+2, at 11:00 (slot 8) — 45min gap
        MeetingRequest(
            duration_slots=2, available_slots=list(range(0, 28)),
            attendees=[0, 1, 2], current_start_slot=8,
        ),
        # Meeting 3: 30min, attendees 0+1, at 13:00 (slot 16) — 1.5hr gap
        MeetingRequest(
            duration_slots=2, available_slots=list(range(0, 28)),
            attendees=[0, 1], current_start_slot=16,
        ),
        # Meeting 4: 30min, attendees 0+2, at 14:00 (slot 20) — 30min gap
        MeetingRequest(
            duration_slots=2, available_slots=list(range(0, 28)),
            attendees=[0, 2], current_start_slot=20,
        ),
    ]

    request = ScheduleRequest(
        total_slots=TOTAL_SLOTS,
        min_focus_block_slots=4,  # 1-hour minimum focus block
        meetings=meetings,
        attendee_free_slots=attendee_free_slots,
        attendee_contexts=attendee_contexts,
        focus_time_constraints=[],
        optimization_mode=OptimizationMode.FOCUS_TIME,
        timeout_seconds=30,
        focus_time_cap=16,
    )

    print("BEFORE (fragmented schedule):")
    print("  A0: M0@09:00, M1@09:45, M2@11:00, M3@13:00, M4@14:00")
    print("  A1: M0@09:00, M2@11:00, M3@13:00")
    print("  A2: M1@09:45, M2@11:00, M4@14:00")
    print("  Many 15-45min gaps — not enough for deep work.")

    solver = Solver(request)
    response = solver.solve()

    print_schedule(request, response, "AFTER DEFRAG")


def example_new_meeting():
    """Schedule a new meeting with minimal disruption to existing calendars."""
    TOTAL_SLOTS = 32

    work_hours = list(range(TOTAL_SLOTS))
    meeting_hours = list(range(4, 28))

    attendee_contexts = [
        AttendeeContext(
            work_hours=work_hours,
            meeting_hours=meeting_hours,
            busy_slots=[4, 5, 6, 7, 16, 17, 18, 19],  # 10:00-11:00 and 13:00-14:00
            tentative_slots=[],
        ),
        AttendeeContext(
            work_hours=work_hours,
            meeting_hours=meeting_hours,
            busy_slots=[8, 9, 10, 11, 20, 21, 22, 23],  # 11:00-12:00 and 14:00-15:00
            tentative_slots=[12, 13],  # 12:00-12:30 tentative
        ),
    ]

    attendee_free_slots = [work_hours[:], work_hours[:]]

    # New meeting: 1hr between attendees 0 and 1
    meetings = [
        MeetingRequest(
            duration_slots=4,
            available_slots=list(range(0, 28)),
            attendees=[0, 1],
            current_start_slot=None,  # New meeting!
        ),
    ]

    request = ScheduleRequest(
        total_slots=TOTAL_SLOTS,
        min_focus_block_slots=4,
        meetings=meetings,
        attendee_free_slots=attendee_free_slots,
        attendee_contexts=attendee_contexts,
        focus_time_constraints=[],
        optimization_mode=OptimizationMode.MIN_DISRUPTION,
        timeout_seconds=10,
    )

    solver = Solver(request)
    solutions = solver.solve_multiple(n=3)

    for idx, sol in enumerate(solutions):
        print_schedule(request, sol, f"OPTION {idx + 1}")


if __name__ == "__main__":
    print("=" * 60)
    print("  EXAMPLE 1: Calendar Defrag")
    print("=" * 60)
    example_defrag()

    print("\n\n")
    print("=" * 60)
    print("  EXAMPLE 2: Schedule New Meeting (Multiple Options)")
    print("=" * 60)
    example_new_meeting()
