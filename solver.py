"""
Meeting Assignment Solver using Integer Linear Programming.

Solves the problem of scheduling/rescheduling meetings across attendees' calendars
to maximize focus time (contiguous free blocks) while minimizing disruption.

Uses Google OR-Tools CP-SAT solver (open source).
"""

from enum import Enum
from typing import Optional

from ortools.sat.python import cp_model

from models import (
    AttendeeContext,
    MeetingRequest,
    MeetingResult,
    OptimizationMode,
    ScheduleRequest,
    ScheduleResponse,
)


class ObjectiveComponent(Enum):
    """Sub-objectives, ordered by priority within each mode."""
    WORK_HOURS_COMPLIANCE = "work_hours_compliance"
    MEETING_HOURS_COMPLIANCE = "meeting_hours_compliance"
    NO_CONFLICT = "no_conflict"
    NO_CONFLICT_TENTATIVE = "no_conflict_tentative"
    TOTAL_NON_OVERLAPPING_TIME = "total_non_overlapping_time"
    STAY_PUT = "stay_put"
    MOVE_COST = "move_cost"
    TIME_ALIGNMENT = "time_alignment"
    FOCUS_TIME = "focus_time"
    FOCUS_TIME_UNCAPPED = "focus_time_uncapped"
    TOTAL_DURATION = "total_duration"
    SCHEDULING_DELAY = "scheduling_delay"


class Solver:
    """ILP-based meeting assignment solver.

    Decision variables:
        s[i,t] : 1 if meeting i starts at slot t
        x[i,t] : 1 if meeting i occupies slot t
        b[j,t] : 1 if attendee j is busy at slot t
        f[j,t] : 1 if attendee j has focus time at slot t
        e[j,t] : edge detector for focus block starts
        c[j]   : capped focus time for attendee j
        d[i]   : duration of meeting i (for dynamic duration)
    """

    def __init__(self, request: ScheduleRequest):
        self.req = request
        self.model = cp_model.CpModel()

        # Decision variables
        self.s: dict[tuple[int, int], cp_model.IntVar] = {}  # start[meeting, slot]
        self.x: dict[tuple[int, int], cp_model.IntVar] = {}  # occupancy[meeting, slot]
        self.b: dict[tuple[int, int], cp_model.IntVar] = {}  # busy[attendee, slot]
        self.f: dict[tuple[int, int], cp_model.IntVar] = {}  # focus[attendee, slot]
        self.e: dict[tuple[int, int], cp_model.IntVar] = {}  # edge[attendee, slot]
        self.c: dict[int, cp_model.IntVar] = {}              # capped_focus[attendee]
        self.d: dict[int, cp_model.IntVar] = {}              # duration[meeting]

        # Objective tracking
        self.objective_exprs: dict[str, cp_model.LinearExpr] = {}
        self.objective_values: dict[str, float] = {}

        self._build_model()

    def _build_model(self):
        """Construct the ILP model."""
        self._create_variables()
        self._add_schedule_all_meetings()
        self._add_meeting_slot_constraints()
        self._add_attendee_busy_constraints()

        if not self.req.simplified:
            self._add_focus_time_constraints()
            self._add_no_regression_constraints()
            self._add_hours_violation_constraints()
            self._add_recurring_meeting_constraints()

        self._build_objective()

    # ------------------------------------------------------------------ #
    #  Variables                                                          #
    # ------------------------------------------------------------------ #

    def _create_variables(self):
        model = self.model

        for i, mtg in enumerate(self.req.meetings):
            avail = set(mtg.effective_available_slots)

            # Start variables: s[i,t] = 1 if meeting i starts at slot t
            for t in avail:
                # Only create if the meeting fits starting at t
                if self._can_start_at(mtg, t):
                    self.s[i, t] = model.new_bool_var(f"s_{i}_{t}")

            # Occupancy variables: x[i,t] = 1 if meeting i occupies slot t
            occupied_slots = set()
            for t in avail:
                if self._can_start_at(mtg, t):
                    for dt in range(mtg.duration_slots):
                        occupied_slots.add(t + dt)
            for t in occupied_slots:
                self.x[i, t] = model.new_bool_var(f"x_{i}_{t}")

            # Dynamic duration variable
            if mtg.is_dynamic_duration:
                self.d[i] = model.new_int_var(
                    mtg.min_duration_slots, mtg.duration_slots, f"d_{i}"
                )

        # Attendee variables
        num_attendees = len(self.req.attendee_free_slots)
        for j in range(num_attendees):
            free_set = set(self.req.attendee_free_slots[j])
            for t in range(self.req.total_slots):
                self.b[j, t] = model.new_bool_var(f"b_{j}_{t}")
                if t in free_set:
                    self.f[j, t] = model.new_bool_var(f"f_{j}_{t}")
                    self.e[j, t] = model.new_bool_var(f"e_{j}_{t}")

            # Capped focus time
            cap = int(self.req.focus_time_cap or self.req.total_slots)
            self.c[j] = model.new_int_var(0, cap, f"c_{j}")

    def _can_start_at(self, mtg: MeetingRequest, t: int) -> bool:
        """Check if meeting can start at slot t (enough consecutive slots)."""
        avail = set(mtg.effective_available_slots)
        for dt in range(mtg.duration_slots):
            slot = t + dt
            if slot >= self.req.total_slots:
                return False
            # For the first slot we need it in available; for subsequent
            # we just need them within bounds (they'll be constrained by occupancy)
        return True

    # ------------------------------------------------------------------ #
    #  Core Constraints                                                   #
    # ------------------------------------------------------------------ #

    def _add_schedule_all_meetings(self):
        """Each meeting must be scheduled exactly once."""
        for i, mtg in enumerate(self.req.meetings):
            start_vars = [self.s[i, t] for t in mtg.effective_available_slots
                          if (i, t) in self.s]
            self.model.add(sum(start_vars) == 1)

    def _add_meeting_slot_constraints(self):
        """Link start variables to occupancy variables.

        If meeting i starts at t, it occupies slots [t, t+duration).
        """
        for i, mtg in enumerate(self.req.meetings):
            for t in mtg.effective_available_slots:
                if (i, t) not in self.s:
                    continue

                # If this meeting starts at t, it must occupy all slots in [t, t+duration)
                for dt in range(mtg.duration_slots):
                    slot = t + dt
                    if (i, slot) in self.x:
                        # s[i,t] implies x[i,slot]
                        self.model.add(self.x[i, slot] >= self.s[i, t])

            # Each occupied slot must be justified by some start
            for t_occ in set(t for (mi, t) in self.x if mi == i):
                # x[i,t_occ] <= sum of s[i,t] for t where t <= t_occ < t + duration
                justifiers = []
                for t_start in mtg.effective_available_slots:
                    if (i, t_start) in self.s:
                        if t_start <= t_occ < t_start + mtg.duration_slots:
                            justifiers.append(self.s[i, t_start])
                if justifiers:
                    self.model.add(self.x[i, t_occ] <= sum(justifiers))

    def _add_attendee_busy_constraints(self):
        """Attendee is busy if any of their meetings occupies that slot."""
        num_attendees = len(self.req.attendee_free_slots)

        for j in range(num_attendees):
            for t in range(self.req.total_slots):
                if (j, t) not in self.b:
                    continue

                # Collect meetings that include this attendee and could occupy slot t
                meeting_vars = []
                for i, mtg in enumerate(self.req.meetings):
                    if j in mtg.attendees and (i, t) in self.x:
                        meeting_vars.append(self.x[i, t])

                # Also mark busy for pre-existing commitments
                ctx = self._get_attendee_context(j)
                if ctx and t in set(ctx.busy_slots):
                    self.model.add(self.b[j, t] == 1)
                elif meeting_vars:
                    # b[j,t] >= x[i,t] for each meeting
                    for mv in meeting_vars:
                        self.model.add(self.b[j, t] >= mv)
                    # b[j,t] <= sum(x[i,t]) — only busy if at least one meeting
                    self.model.add(self.b[j, t] <= sum(meeting_vars))
                    # No overlap: at most one meeting per attendee per slot
                    self.model.add(sum(meeting_vars) <= 1)

    # ------------------------------------------------------------------ #
    #  Focus Time Constraints                                             #
    # ------------------------------------------------------------------ #

    def _add_focus_time_constraints(self):
        """Focus time = slots in contiguous free blocks >= min_focus_block_slots.

        A slot counts as "focus time" only if it's part of a contiguous free
        block of at least `min_block` slots. We use auxiliary variables:
          g[j,t] = 1 if slot t is "quality focus time" for attendee j

        For g[j,t] = 1 we require that all slots in some window of min_block
        containing t are free. We model this by: for each possible window
        start w, create a variable that's 1 iff all min_block slots in
        [w, w+min_block) are free. Then g[j,t] = OR of all windows containing t.
        """
        num_attendees = len(self.req.attendee_free_slots)
        min_block = self.req.min_focus_block_slots

        for j in range(num_attendees):
            free_slots = sorted(self.req.attendee_free_slots[j])
            free_set = set(free_slots)

            # f[j,t] = 1 iff attendee j is free at slot t
            for t in free_slots:
                if (j, t) in self.f and (j, t) in self.b:
                    self.model.add(self.f[j, t] + self.b[j, t] == 1)

            # g[j,t] = 1 if slot t is in a contiguous free block >= min_block
            g: dict[int, cp_model.IntVar] = {}

            # Find all possible window starts: w such that [w, w+min_block) ⊆ free_set
            valid_windows: list[int] = []
            for w in free_slots:
                if all((w + d) in free_set for d in range(min_block)):
                    valid_windows.append(w)

            if not valid_windows:
                self.model.add(self.c[j] == 0)
                continue

            # w_var[w] = 1 iff ALL slots in [w, w+min_block) are free
            w_var: dict[int, cp_model.IntVar] = {}
            for w in valid_windows:
                wv = self.model.new_bool_var(f"w_{j}_{w}")
                w_var[w] = wv
                window_free = [self.f[j, w + d] for d in range(min_block)
                               if (j, w + d) in self.f]
                # wv = 1 implies all are free; wv = 0 if any is busy
                for fv in window_free:
                    self.model.add(wv <= fv)
                # wv >= sum(free) - (min_block - 1)  [all free implies wv can be 1]
                self.model.add(wv >= sum(window_free) - (min_block - 1))

            # g[j,t] = 1 if any window containing t has all free slots
            for t in free_slots:
                if (j, t) not in self.f:
                    continue
                # Windows that contain t: w where w <= t < w + min_block
                covering = [w_var[w] for w in valid_windows
                            if w <= t < w + min_block]
                if covering:
                    gv = self.model.new_bool_var(f"g_{j}_{t}")
                    g[t] = gv
                    # g <= f (must be free)
                    self.model.add(gv <= self.f[j, t])
                    # g <= sum(covering windows)
                    self.model.add(gv <= sum(covering))
                    # g >= each covering window (if any window is valid, g can be 1)
                    for cv in covering:
                        self.model.add(gv >= cv)

            # Capped focus time = min(sum of g[j,t], cap)
            if g:
                total_focus = sum(g.values())
                self.model.add(self.c[j] <= total_focus)
            else:
                self.model.add(self.c[j] == 0)

    def _add_no_regression_constraints(self):
        """Ensure focus time doesn't decrease from current schedule."""
        for ftc in self.req.focus_time_constraints:
            j = ftc.attendee_index
            if j in self.c:
                min_ft = int(ftc.min_focus_time)
                self.model.add(self.c[j] >= min_ft)

    # ------------------------------------------------------------------ #
    #  Hours Violation Constraints                                        #
    # ------------------------------------------------------------------ #

    def _add_hours_violation_constraints(self):
        """Track meetings scheduled outside work/meeting hours."""
        for i, mtg in enumerate(self.req.meetings):
            for j in mtg.attendees:
                ctx = self._get_attendee_context(j)
                if not ctx:
                    continue

                work_set = set(ctx.work_hours)
                meeting_set = set(ctx.meeting_hours)

                # Work hours violation: slots where meeting is scheduled outside work hours
                work_violations = []
                for t in range(self.req.total_slots):
                    if (i, t) in self.x and t not in work_set:
                        work_violations.append(self.x[i, t])

                if work_violations:
                    key = f"whv_{i}_{j}"
                    whv = self.model.new_int_var(0, len(work_violations), key)
                    self.model.add(whv == sum(work_violations))

                # Meeting hours violation
                mtg_violations = []
                for t in range(self.req.total_slots):
                    if (i, t) in self.x and t not in meeting_set:
                        mtg_violations.append(self.x[i, t])

                if mtg_violations:
                    key = f"mhv_{i}_{j}"
                    mhv = self.model.new_int_var(0, len(mtg_violations), key)
                    self.model.add(mhv == sum(mtg_violations))

    # ------------------------------------------------------------------ #
    #  Recurring Meeting Constraints                                      #
    # ------------------------------------------------------------------ #

    def _add_recurring_meeting_constraints(self):
        """Recurring meeting instances must maintain their relative day offsets."""
        # Group meetings by having occurrence_day_offset set
        recurring = [
            (i, mtg) for i, mtg in enumerate(self.req.meetings)
            if mtg.occurrence_day_offset is not None
        ]

        if len(recurring) < 2:
            return

        # Each pair of recurring meetings must maintain their slot offset
        base_i, base_mtg = recurring[0]
        for other_i, other_mtg in recurring[1:]:
            offset = other_mtg.occurrence_day_offset - base_mtg.occurrence_day_offset

            # Sum(s[base,t]*t) + offset == Sum(s[other,t]*t)
            base_start = sum(
                self.s[base_i, t] * t
                for t in base_mtg.effective_available_slots
                if (base_i, t) in self.s
            )
            other_start = sum(
                self.s[other_i, t] * t
                for t in other_mtg.effective_available_slots
                if (other_i, t) in self.s
            )
            self.model.add(other_start - base_start == offset)

    # ------------------------------------------------------------------ #
    #  Objective                                                          #
    # ------------------------------------------------------------------ #

    def _build_objective(self):
        """Build weighted multi-objective function.

        Uses large weight multipliers to enforce lexicographic priority ordering.
        """
        mode = self.req.optimization_mode
        weights = self.req.objective_weights

        objectives: list[tuple[str, cp_model.LinearExpr, int, bool]] = []
        # (name, expression, priority_weight, maximize)

        if mode == OptimizationMode.FOCUS_TIME:
            objectives = self._focus_time_objectives()
        else:
            objectives = self._min_disruption_objectives()

        # Build combined objective with lexicographic weighting
        total_obj = 0
        for name, expr, weight, maximize in objectives:
            user_weight = weights.get(name, 1.0)
            scaled_weight = int(weight * user_weight)
            if maximize:
                total_obj += scaled_weight * expr
            else:
                total_obj -= scaled_weight * expr

        self.model.maximize(total_obj)

    def _focus_time_objectives(self) -> list:
        """Objectives for calendar defrag mode."""
        T = self.req.total_slots
        N = len(self.req.meetings)
        objectives = []

        # Priority 1 (highest): Maximize capped focus time
        focus_expr = sum(self.c[j] for j in self.c)
        objectives.append(("focus_time", focus_expr, T * N * 100, True))

        # Priority 2: Minimize conflicts for new meetings
        conflict_expr = self._conflict_expression()
        if conflict_expr is not None:
            objectives.append(("no_conflict", conflict_expr, T * N * 10, False))

        # Priority 3: Prefer slots in preferred_slots (e.g., in-person eligible)
        pref_expr = self._preferred_slots_expression()
        if pref_expr is not None:
            objectives.append(("preferred_slots", pref_expr, T * N, True))

        # Priority 4: Time alignment (prefer :00 and :30 starts)
        align_expr = self._time_alignment_expression()
        if align_expr is not None:
            objectives.append(("time_alignment", align_expr, N, True))

        # Priority 5: Stay put (minimize number of moved meetings)
        stay_expr = self._stay_put_expression()
        if stay_expr is not None:
            objectives.append(("stay_put", stay_expr, 1, True))

        return objectives

    def _min_disruption_objectives(self) -> list:
        """Objectives for meeting scheduling mode."""
        T = self.req.total_slots
        N = len(self.req.meetings)
        objectives = []

        # Priority 1: Meeting hours compliance
        mhv = self._meeting_hours_violation_expression()
        if mhv is not None:
            objectives.append(("meeting_hours_compliance", mhv, T**2 * N**2 * 1000, False))

        # Priority 2: Work hours compliance
        whv = self._work_hours_violation_expression()
        if whv is not None:
            objectives.append(("work_hours_compliance", whv, T**2 * N**2 * 100, False))

        # Priority 3: No conflict (busy)
        conflict_expr = self._conflict_expression()
        if conflict_expr is not None:
            objectives.append(("no_conflict", conflict_expr, T**2 * N * 10, False))

        # Priority 4: No conflict (tentative)
        tent_expr = self._tentative_conflict_expression()
        if tent_expr is not None:
            objectives.append(("no_conflict_tentative", tent_expr, T**2 * N, False))

        # Priority 5: Stay put
        stay_expr = self._stay_put_expression()
        if stay_expr is not None:
            objectives.append(("stay_put", stay_expr, T * N, True))

        # Priority 6: Move cost (quadratic penalty)
        move_expr = self._move_cost_expression()
        if move_expr is not None:
            objectives.append(("move_cost", move_expr, T, False))

        # Priority 7: Time alignment
        align_expr = self._time_alignment_expression()
        if align_expr is not None:
            objectives.append(("time_alignment", align_expr, 1, True))

        # Priority 8: Focus time (if available)
        if self.c:
            focus_expr = sum(self.c[j] for j in self.c)
            objectives.append(("focus_time", focus_expr, 1, True))

        return objectives

    # ---- Objective helpers ---- #

    def _stay_put_expression(self):
        """Count meetings that stay in their current slot."""
        stay_vars = []
        for i, mtg in enumerate(self.req.meetings):
            if mtg.current_start_slot is not None and (i, mtg.current_start_slot) in self.s:
                stay_vars.append(self.s[i, mtg.current_start_slot])
        return sum(stay_vars) if stay_vars else None

    def _move_cost_expression(self):
        """Weighted squared distance from current slot."""
        costs = []
        for i, mtg in enumerate(self.req.meetings):
            if mtg.current_start_slot is None:
                continue
            for t in mtg.effective_available_slots:
                if (i, t) not in self.s:
                    continue
                dist = abs(t - mtg.current_start_slot)
                costs.append(self.s[i, t] * int(mtg.move_penalty * dist * dist))
        return sum(costs) if costs else None

    def _time_alignment_expression(self):
        """Prefer starts on the hour (:00) or half-hour (:30)."""
        alignment_vars = []
        slots_per_hour = 60 // self.req.slot_duration_minutes
        for i, mtg in enumerate(self.req.meetings):
            for t in mtg.effective_available_slots:
                if (i, t) not in self.s:
                    continue
                # Slots 0, 2, 4, ... are on the hour/half-hour (for 15-min slots)
                if t % (slots_per_hour // 2) == 0:
                    alignment_vars.append(self.s[i, t] * 2)
                elif t % (slots_per_hour // 4) == 0 if slots_per_hour >= 4 else False:
                    alignment_vars.append(self.s[i, t] * 1)
        return sum(alignment_vars) if alignment_vars else None

    def _conflict_expression(self):
        """Count conflicts with busy slots for new meetings."""
        conflicts = []
        for i, mtg in enumerate(self.req.meetings):
            if mtg.current_start_slot is not None:
                continue  # Only for new meetings
            for j in mtg.attendees:
                ctx = self._get_attendee_context(j)
                if not ctx:
                    continue
                busy = set(ctx.busy_slots)
                for t in range(self.req.total_slots):
                    if (i, t) in self.x and t in busy:
                        conflicts.append(self.x[i, t])
        return sum(conflicts) if conflicts else None

    def _tentative_conflict_expression(self):
        """Count conflicts with tentative slots."""
        conflicts = []
        for i, mtg in enumerate(self.req.meetings):
            if mtg.current_start_slot is not None:
                continue
            for j in mtg.attendees:
                ctx = self._get_attendee_context(j)
                if not ctx:
                    continue
                tent = set(ctx.tentative_slots)
                for t in range(self.req.total_slots):
                    if (i, t) in self.x and t in tent:
                        conflicts.append(self.x[i, t])
        return sum(conflicts) if conflicts else None

    def _meeting_hours_violation_expression(self):
        """Total meeting hours violations across all meetings/attendees."""
        violations = []
        for i, mtg in enumerate(self.req.meetings):
            for j in mtg.attendees:
                ctx = self._get_attendee_context(j)
                if not ctx or not ctx.meeting_hours:
                    continue
                mh = set(ctx.meeting_hours)
                for t in range(self.req.total_slots):
                    if (i, t) in self.x and t not in mh:
                        violations.append(self.x[i, t])
        return sum(violations) if violations else None

    def _work_hours_violation_expression(self):
        """Total work hours violations."""
        violations = []
        for i, mtg in enumerate(self.req.meetings):
            for j in mtg.attendees:
                ctx = self._get_attendee_context(j)
                if not ctx or not ctx.work_hours:
                    continue
                wh = set(ctx.work_hours)
                for t in range(self.req.total_slots):
                    if (i, t) in self.x and t not in wh:
                        violations.append(self.x[i, t])
        return sum(violations) if violations else None

    def _preferred_slots_expression(self):
        """Count meetings scheduled in preferred slots."""
        pref_vars = []
        for i, mtg in enumerate(self.req.meetings):
            if not mtg.preferred_slots:
                continue
            pref = set(mtg.preferred_slots)
            for t in mtg.effective_available_slots:
                if (i, t) in self.s and t in pref:
                    pref_vars.append(self.s[i, t])
        return sum(pref_vars) if pref_vars else None

    # ------------------------------------------------------------------ #
    #  Solve                                                              #
    # ------------------------------------------------------------------ #

    def solve(self) -> ScheduleResponse:
        """Solve the ILP and return the schedule."""
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.req.timeout_seconds

        status = solver.solve(self.model)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return ScheduleResponse(
                changed_meetings={},
                status="INFEASIBLE",
            )

        # Extract solution
        changed = {}
        for i, mtg in enumerate(self.req.meetings):
            for t in mtg.effective_available_slots:
                if (i, t) in self.s and solver.value(self.s[i, t]) == 1:
                    dur = mtg.duration_slots
                    if i in self.d:
                        dur = solver.value(self.d[i])

                    # Only report if changed or new
                    if mtg.current_start_slot is None or t != mtg.current_start_slot or dur != mtg.duration_slots:
                        changed[i] = MeetingResult(start_slot=t, duration_slots=dur)
                    break

        status_str = "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE"
        return ScheduleResponse(
            changed_meetings=changed,
            objective_values={"objective": solver.objective_value},
            status=status_str,
        )

    def solve_multiple(self, n: int = 3) -> list[ScheduleResponse]:
        """Generate N alternative solutions by iteratively excluding previous assignments.

        Useful for offering the user multiple scheduling options.
        """
        solutions = []
        new_meeting_indices = [
            i for i, mtg in enumerate(self.req.meetings)
            if mtg.current_start_slot is None
        ]

        if not new_meeting_indices:
            # No new meetings — just solve once
            return [self.solve()]

        blocked_slots: list[tuple[int, int]] = []  # (meeting_idx, slot)

        for _ in range(n):
            # Block previously found assignments
            for mi, slot in blocked_slots:
                if (mi, slot) in self.s:
                    self.model.add(self.s[mi, slot] == 0)

            response = self.solve()
            if response.status == "INFEASIBLE":
                break

            solutions.append(response)

            # Record the new meeting's assigned slot to block it next iteration
            for mi in new_meeting_indices:
                if mi in response.changed_meetings:
                    blocked_slots.append((mi, response.changed_meetings[mi].start_slot))

        return solutions

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _get_attendee_context(self, j: int) -> Optional[AttendeeContext]:
        if j < len(self.req.attendee_contexts):
            return self.req.attendee_contexts[j]
        return None
