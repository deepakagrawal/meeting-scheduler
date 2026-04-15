"""Tests for the meeting assignment solver."""

import unittest

from models import (
    AttendeeContext,
    FocusTimeConstraint,
    MeetingRequest,
    MeetingResult,
    OptimizationMode,
    ScheduleRequest,
    ScheduleResponse,
)
from solver import Solver


class TestBasicScheduling(unittest.TestCase):
    """Test that meetings get scheduled correctly."""

    def _make_request(self, **kwargs) -> ScheduleRequest:
        defaults = dict(
            total_slots=32,
            min_focus_block_slots=4,
            meetings=[],
            attendee_free_slots=[],
            attendee_contexts=[],
            focus_time_constraints=[],
            optimization_mode=OptimizationMode.FOCUS_TIME,
            timeout_seconds=30,
        )
        defaults.update(kwargs)
        return ScheduleRequest(**defaults)

    def test_single_new_meeting(self):
        """A single new meeting should be scheduled."""
        req = self._make_request(
            meetings=[
                MeetingRequest(
                    duration_slots=2,
                    available_slots=[0, 1, 2, 3, 4],
                    attendees=[0],
                ),
            ],
            attendee_free_slots=[list(range(32))],
            attendee_contexts=[AttendeeContext(work_hours=list(range(32)))],
        )
        solver = Solver(req)
        resp = solver.solve()

        self.assertEqual(resp.status, "OPTIMAL")
        self.assertIn(0, resp.changed_meetings)
        result = resp.changed_meetings[0]
        self.assertEqual(result.duration_slots, 2)
        self.assertIn(result.start_slot, [0, 1, 2, 3, 4])

    def test_meeting_stays_put(self):
        """An existing meeting with no benefit to move should stay put."""
        req = self._make_request(
            meetings=[
                MeetingRequest(
                    duration_slots=2,
                    available_slots=list(range(10)),
                    attendees=[0],
                    current_start_slot=4,
                ),
            ],
            attendee_free_slots=[list(range(32))],
            attendee_contexts=[AttendeeContext(work_hours=list(range(32)))],
        )
        solver = Solver(req)
        resp = solver.solve()

        self.assertEqual(resp.status, "OPTIMAL")
        # Should have no changes (meeting stays at slot 4)
        self.assertEqual(len(resp.changed_meetings), 0)

    def test_two_meetings_no_overlap(self):
        """Two meetings with the same attendee must not overlap."""
        req = self._make_request(
            meetings=[
                MeetingRequest(
                    duration_slots=4,
                    available_slots=list(range(8)),
                    attendees=[0],
                ),
                MeetingRequest(
                    duration_slots=4,
                    available_slots=list(range(8)),
                    attendees=[0],
                ),
            ],
            attendee_free_slots=[list(range(32))],
            attendee_contexts=[AttendeeContext(work_hours=list(range(32)))],
        )
        solver = Solver(req)
        resp = solver.solve()

        self.assertEqual(resp.status, "OPTIMAL")
        self.assertIn(0, resp.changed_meetings)
        self.assertIn(1, resp.changed_meetings)

        r0 = resp.changed_meetings[0]
        r1 = resp.changed_meetings[1]
        slots_0 = set(range(r0.start_slot, r0.start_slot + r0.duration_slots))
        slots_1 = set(range(r1.start_slot, r1.start_slot + r1.duration_slots))
        self.assertEqual(len(slots_0 & slots_1), 0, "Meetings must not overlap")

    def test_infeasible(self):
        """Two 4-slot meetings in a 4-slot window with same attendee = infeasible."""
        req = self._make_request(
            total_slots=4,
            meetings=[
                MeetingRequest(
                    duration_slots=4,
                    available_slots=[0],
                    attendees=[0],
                ),
                MeetingRequest(
                    duration_slots=4,
                    available_slots=[0],
                    attendees=[0],
                ),
            ],
            attendee_free_slots=[list(range(4))],
            attendee_contexts=[AttendeeContext(work_hours=list(range(4)))],
        )
        solver = Solver(req)
        resp = solver.solve()

        self.assertEqual(resp.status, "INFEASIBLE")


class TestFocusTime(unittest.TestCase):
    """Test focus time optimization."""

    def test_defrag_creates_focus_blocks(self):
        """Solver should produce a schedule with valid focus blocks.

        Three 2-slot meetings for a single attendee in a 24-slot day.
        After solving, meetings must not overlap and at least one contiguous
        free block >= min_focus_block_slots (4) should exist.
        """
        total = 24
        work_hours = list(range(total))

        req = ScheduleRequest(
            total_slots=total,
            min_focus_block_slots=4,
            meetings=[
                MeetingRequest(
                    duration_slots=2, available_slots=list(range(20)),
                    attendees=[0], current_start_slot=0,
                ),
                MeetingRequest(
                    duration_slots=2, available_slots=list(range(20)),
                    attendees=[0], current_start_slot=5,
                ),
                MeetingRequest(
                    duration_slots=2, available_slots=list(range(20)),
                    attendees=[0], current_start_slot=10,
                ),
            ],
            attendee_free_slots=[work_hours],
            attendee_contexts=[AttendeeContext(work_hours=work_hours, meeting_hours=work_hours)],
            focus_time_constraints=[],
            optimization_mode=OptimizationMode.FOCUS_TIME,
            timeout_seconds=10,
            focus_time_cap=total,
        )

        solver = Solver(req)
        resp = solver.solve()
        self.assertEqual(resp.status, "OPTIMAL")

        # Get final positions
        positions = []
        for i in range(3):
            if i in resp.changed_meetings:
                positions.append(resp.changed_meetings[i].start_slot)
            else:
                positions.append(req.meetings[i].current_start_slot)

        # Meetings must not overlap
        occupied = set()
        for pos in positions:
            for dt in range(2):
                self.assertNotIn(pos + dt, occupied, "Meetings must not overlap")
                occupied.add(pos + dt)

        # There should be at least one free block >= 4 slots
        free_slots = sorted(set(range(total)) - occupied)
        max_block = 0
        current_block = 0
        prev = -2
        for s in free_slots:
            if s == prev + 1:
                current_block += 1
            else:
                current_block = 1
            max_block = max(max_block, current_block)
            prev = s
        self.assertGreaterEqual(max_block, 4,
                                "Should create at least one focus block >= 4 slots")

    def test_no_regression(self):
        """Focus time should not decrease if no_regression constraint is set."""
        total = 16
        work_hours = list(range(total))

        req = ScheduleRequest(
            total_slots=total,
            min_focus_block_slots=4,
            meetings=[
                MeetingRequest(
                    duration_slots=2,
                    available_slots=list(range(12)),
                    attendees=[0],
                    current_start_slot=0,
                ),
            ],
            attendee_free_slots=[work_hours],
            attendee_contexts=[AttendeeContext(work_hours=work_hours)],
            focus_time_constraints=[
                FocusTimeConstraint(attendee_index=0, min_focus_time=8),
            ],
            optimization_mode=OptimizationMode.FOCUS_TIME,
            timeout_seconds=10,
            focus_time_cap=total,
        )

        solver = Solver(req)
        resp = solver.solve()
        self.assertEqual(resp.status, "OPTIMAL")


class TestMinDisruption(unittest.TestCase):
    """Test the MIN_DISRUPTION optimization mode."""

    def test_avoids_busy_slots(self):
        """New meeting should avoid attendee's busy slots."""
        total = 16
        work_hours = list(range(total))

        req = ScheduleRequest(
            total_slots=total,
            min_focus_block_slots=4,
            meetings=[
                MeetingRequest(
                    duration_slots=4,
                    available_slots=list(range(12)),
                    attendees=[0],
                ),
            ],
            attendee_free_slots=[work_hours],
            attendee_contexts=[
                AttendeeContext(
                    work_hours=work_hours,
                    meeting_hours=work_hours,
                    busy_slots=[0, 1, 2, 3],  # First hour busy
                ),
            ],
            focus_time_constraints=[],
            optimization_mode=OptimizationMode.MIN_DISRUPTION,
            timeout_seconds=10,
        )

        solver = Solver(req)
        resp = solver.solve()
        self.assertEqual(resp.status, "OPTIMAL")
        self.assertIn(0, resp.changed_meetings)

        start = resp.changed_meetings[0].start_slot
        meeting_slots = set(range(start, start + 4))
        busy = {0, 1, 2, 3}
        self.assertEqual(len(meeting_slots & busy), 0, "Should not overlap busy slots")

    def test_prefers_meeting_hours(self):
        """New meeting should prefer meeting hours over non-meeting hours."""
        total = 20
        work_hours = list(range(total))
        meeting_hours = list(range(4, 16))  # 10:00-13:00

        req = ScheduleRequest(
            total_slots=total,
            min_focus_block_slots=4,
            meetings=[
                MeetingRequest(
                    duration_slots=4,
                    available_slots=list(range(16)),
                    attendees=[0],
                ),
            ],
            attendee_free_slots=[work_hours],
            attendee_contexts=[
                AttendeeContext(
                    work_hours=work_hours,
                    meeting_hours=meeting_hours,
                ),
            ],
            focus_time_constraints=[],
            optimization_mode=OptimizationMode.MIN_DISRUPTION,
            timeout_seconds=10,
        )

        solver = Solver(req)
        resp = solver.solve()
        self.assertEqual(resp.status, "OPTIMAL")
        start = resp.changed_meetings[0].start_slot
        # All occupied slots should be within meeting hours
        for dt in range(4):
            self.assertIn(start + dt, meeting_hours,
                          f"Slot {start + dt} should be within meeting hours")


class TestMultipleSolutions(unittest.TestCase):
    """Test solve_multiple returns distinct solutions."""

    def test_multiple_options(self):
        """solve_multiple should return different start times."""
        total = 20
        work_hours = list(range(total))

        req = ScheduleRequest(
            total_slots=total,
            min_focus_block_slots=4,
            meetings=[
                MeetingRequest(
                    duration_slots=2,
                    available_slots=list(range(16)),
                    attendees=[0],
                ),
            ],
            attendee_free_slots=[work_hours],
            attendee_contexts=[AttendeeContext(work_hours=work_hours)],
            focus_time_constraints=[],
            optimization_mode=OptimizationMode.MIN_DISRUPTION,
            timeout_seconds=10,
        )

        solver = Solver(req)
        solutions = solver.solve_multiple(n=3)

        self.assertGreaterEqual(len(solutions), 2, "Should find at least 2 solutions")

        starts = [s.changed_meetings[0].start_slot for s in solutions]
        self.assertEqual(len(starts), len(set(starts)), "Each solution should have a unique start time")


class TestRecurringMeetings(unittest.TestCase):
    """Test recurring meeting constraints."""

    def test_recurring_maintains_offset(self):
        """Recurring instances must maintain their day offset."""
        total = 32
        work_hours = list(range(total))

        # Two instances of a recurring meeting, 8 slots apart (2 hours)
        req = ScheduleRequest(
            total_slots=total,
            min_focus_block_slots=4,
            meetings=[
                MeetingRequest(
                    duration_slots=2,
                    available_slots=list(range(24)),
                    attendees=[0],
                    current_start_slot=0,
                    occurrence_day_offset=0,
                ),
                MeetingRequest(
                    duration_slots=2,
                    available_slots=list(range(24)),
                    attendees=[0],
                    current_start_slot=8,
                    occurrence_day_offset=8,
                ),
            ],
            attendee_free_slots=[work_hours],
            attendee_contexts=[AttendeeContext(work_hours=work_hours)],
            focus_time_constraints=[],
            optimization_mode=OptimizationMode.FOCUS_TIME,
            timeout_seconds=10,
            focus_time_cap=total,
        )

        solver = Solver(req)
        resp = solver.solve()
        self.assertEqual(resp.status, "OPTIMAL")

        # Get final positions
        s0 = resp.changed_meetings[0].start_slot if 0 in resp.changed_meetings else 0
        s1 = resp.changed_meetings[1].start_slot if 1 in resp.changed_meetings else 8

        self.assertEqual(s1 - s0, 8, "Recurring meetings must maintain 8-slot offset")


class TestMultiAttendee(unittest.TestCase):
    """Test scenarios with multiple attendees."""

    def test_respects_all_attendee_busy_slots(self):
        """Meeting must avoid busy slots of ALL attendees."""
        total = 16
        work_hours = list(range(total))

        req = ScheduleRequest(
            total_slots=total,
            min_focus_block_slots=4,
            meetings=[
                MeetingRequest(
                    duration_slots=4,
                    available_slots=list(range(12)),
                    attendees=[0, 1],
                ),
            ],
            attendee_free_slots=[work_hours, work_hours],
            attendee_contexts=[
                AttendeeContext(work_hours=work_hours, meeting_hours=work_hours,
                                busy_slots=[0, 1, 2, 3]),
                AttendeeContext(work_hours=work_hours, meeting_hours=work_hours,
                                busy_slots=[4, 5, 6, 7]),
            ],
            focus_time_constraints=[],
            optimization_mode=OptimizationMode.MIN_DISRUPTION,
            timeout_seconds=10,
        )

        solver = Solver(req)
        resp = solver.solve()
        self.assertEqual(resp.status, "OPTIMAL")

        start = resp.changed_meetings[0].start_slot
        meeting_slots = set(range(start, start + 4))
        all_busy = {0, 1, 2, 3, 4, 5, 6, 7}
        self.assertEqual(len(meeting_slots & all_busy), 0,
                         "Must avoid busy slots of both attendees")


class TestEdgeCases(unittest.TestCase):
    """Edge cases and boundary conditions."""

    def test_single_slot_meeting(self):
        """A 1-slot (15min) meeting should work."""
        req = ScheduleRequest(
            total_slots=8,
            min_focus_block_slots=4,
            meetings=[
                MeetingRequest(duration_slots=1, available_slots=[0, 1, 2], attendees=[0]),
            ],
            attendee_free_slots=[list(range(8))],
            attendee_contexts=[AttendeeContext(work_hours=list(range(8)))],
            optimization_mode=OptimizationMode.MIN_DISRUPTION,
            timeout_seconds=10,
        )
        solver = Solver(req)
        resp = solver.solve()
        self.assertEqual(resp.status, "OPTIMAL")
        self.assertEqual(resp.changed_meetings[0].duration_slots, 1)

    def test_no_meetings(self):
        """Empty meeting list should return no changes."""
        req = ScheduleRequest(
            total_slots=8,
            min_focus_block_slots=4,
            meetings=[],
            attendee_free_slots=[list(range(8))],
            attendee_contexts=[AttendeeContext(work_hours=list(range(8)))],
            optimization_mode=OptimizationMode.FOCUS_TIME,
            timeout_seconds=10,
        )
        solver = Solver(req)
        resp = solver.solve()
        self.assertEqual(len(resp.changed_meetings), 0)

    def test_meeting_fills_entire_horizon(self):
        """A meeting that takes the entire time horizon."""
        req = ScheduleRequest(
            total_slots=4,
            min_focus_block_slots=4,
            meetings=[
                MeetingRequest(duration_slots=4, available_slots=[0], attendees=[0]),
            ],
            attendee_free_slots=[list(range(4))],
            attendee_contexts=[AttendeeContext(work_hours=list(range(4)))],
            optimization_mode=OptimizationMode.MIN_DISRUPTION,
            timeout_seconds=10,
        )
        solver = Solver(req)
        resp = solver.solve()
        self.assertEqual(resp.status, "OPTIMAL")
        self.assertEqual(resp.changed_meetings[0].start_slot, 0)
        self.assertEqual(resp.changed_meetings[0].duration_slots, 4)


if __name__ == "__main__":
    unittest.main()
