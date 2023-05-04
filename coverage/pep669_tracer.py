# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/nedbat/coveragepy/blob/master/NOTICE.txt

"""Raw data collector for coverage.py."""

from __future__ import annotations

import atexit
import inspect
import sys
import threading
import traceback

from types import CodeType, FrameType, ModuleType
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, cast

from coverage import env
from coverage.types import (
    TArc, TFileDisposition, TLineNo, TTraceData, TTraceFileData, TTraceFn,
    TTracer, TWarnFn,
)

# When running meta-coverage, this file can try to trace itself, which confuses
# everything.  Don't trace ourselves.

THIS_FILE = __file__.rstrip("co")


def log(msg):
    with open("/tmp/pan.out", "a") as f:
        print(msg, file=f)

def panopticon(meth):
    def _wrapped(self, *args, **kwargs):
        assert not kwargs
        log(f"{meth.__name__}{args!r}")
        try:
            return meth(self, *args, **kwargs)
        except:
            with open("/tmp/pan.out", "a") as f:
                traceback.print_exception(sys.exception(), file=f)
                sys.monitoring.set_events(sys.monitoring.COVERAGE_ID, 0)
            raise
    return _wrapped


class Pep669Tracer(TTracer):
    """Python implementation of the raw data tracer for PEP669 implementations."""

    def __init__(self) -> None:
        # pylint: disable=super-init-not-called
        # Attributes set from the collector:
        self.data: TTraceData
        self.trace_arcs = False
        self.should_trace: Callable[[str, FrameType], TFileDisposition]
        self.should_trace_cache: Dict[str, Optional[TFileDisposition]]
        self.should_start_context: Optional[Callable[[FrameType], Optional[str]]] = None
        self.switch_context: Optional[Callable[[Optional[str]], None]] = None
        self.warn: TWarnFn

        # The threading module to use, if any.
        self.threading: Optional[ModuleType] = None

        self.cur_file_data: Optional[TTraceFileData] = None
        self.last_line: TLineNo = 0
        self.cur_file_name: Optional[str] = None
        #self.context: Optional[str] = None

        self.code_cache: Dict[CodeType, Tuple[bool, Optional[TTraceFileData]]] = {}

        # The frame_stack parallels the Python call stack. Each entry is
        # information about an active frame, a three-element tuple:
        #   [0] The TTraceData for this frame's file. Could be None if we
        #           aren't tracing this frame.
        #   [1] The current file name for the frame. None if we aren't tracing
        #           this frame.
        #   [2] The last line number executed in this frame.
        self.frame_stack: List[Tuple[Optional[TTraceFileData], Optional[str], TLineNo]] = []
        self.thread: Optional[threading.Thread] = None
        self.stopped = False
        self._activity = False

        self.in_atexit = False
        # On exit, self.in_atexit = True
        atexit.register(setattr, self, "in_atexit", True)

    def __repr__(self) -> str:
        me = id(self)
        points = sum(len(v) for v in self.data.values())
        files = len(self.data)
        return f"<Pep669Tracer at 0x{me:x}: {points} data points in {files} files>"

    def log(self, marker: str, *args: Any) -> None:
        """For hard-core logging of what this tracer is doing."""
        with open("/tmp/debug_trace.txt", "a") as f:
            f.write("{} {}[{}]".format(
                marker,
                id(self),
                len(self.frame_stack),
            ))
            if 0:   # if you want thread ids..
                f.write(".{:x}.{:x}".format(                    # type: ignore[unreachable]
                    self.thread.ident,
                    self.threading.current_thread().ident,
                ))
            f.write(" {}".format(" ".join(map(str, args))))
            if 0:   # if you want callers..
                f.write(" | ")                                  # type: ignore[unreachable]
                stack = " / ".join(
                    (fname or "???").rpartition("/")[-1]
                    for _, fname, _, _ in self.frame_stack
                )
                f.write(stack)
            f.write("\n")

    def start(self) -> TTraceFn:
        """Start this Tracer.

        Return a Python function suitable for use with sys.settrace().

        """
        self.stopped = False
        if self.threading:
            if self.thread is None:
                self.thread = self.threading.current_thread()
            else:
                if self.thread.ident != self.threading.current_thread().ident:
                    # Re-starting from a different thread!? Don't set the trace
                    # function, but we are marked as running again, so maybe it
                    # will be ok?
                    #self.log("~", "starting on different threads")
                    return self._cached_bound_method_trace

        self.myid = sys.monitoring.COVERAGE_ID
        sys.monitoring.use_tool_id(self.myid, "coverage.py")
        events = sys.monitoring.events
        sys.monitoring.set_events(self.myid, events.PY_START)
        sys.monitoring.register_callback(self.myid, events.PY_START, self.sysmon_py_start)
        # Use PY_START globally, then use set_local_event(LINE) for interesting
        # frames, so i might not need to bookkeep which are the interesting frame.
        sys.monitoring.register_callback(self.myid, events.PY_RESUME, self.sysmon_py_resume)
        sys.monitoring.register_callback(self.myid, events.PY_RETURN, self.sysmon_py_return)
        sys.monitoring.register_callback(self.myid, events.PY_YIELD, self.sysmon_py_yield)
        # UNWIND is like RETURN/YIELD
        sys.monitoring.register_callback(self.myid, events.LINE, self.sysmon_line)
        #sys.monitoring.register_callback(self.myid, events.BRANCH, self.sysmon_branch)

    def stop(self) -> None:
        """Stop this Tracer."""
        sys.monitoring.set_events(self.myid, 0)
        sys.monitoring.free_tool_id(self.myid)

    def activity(self) -> bool:
        """Has there been any activity?"""
        return self._activity

    def reset_activity(self) -> None:
        """Reset the activity() flag."""
        self._activity = False

    def get_stats(self) -> Optional[Dict[str, int]]:
        """Return a dictionary of statistics, or None."""
        return None

    @panopticon
    def sysmon_py_start(self, code, instruction_offset: int):
        # Entering a new frame.  Decide if we should trace in this file.
        self._activity = True

        self.frame_stack.append((self.cur_file_data, self.cur_file_name, self.last_line))

        if code in self.code_cache:
            tracing_code, self.cur_file_data = self.code_cache[code]
        else:
            tracing_code = self.cur_file_data = None

        if tracing_code is None:
            self.cur_file_name = filename = code.co_filename
            disp = self.should_trace_cache.get(filename)
            if disp is None:
                frame = inspect.currentframe()
                disp = self.should_trace(filename, frame)
                self.should_trace_cache[filename] = disp

            tracing_code = disp.trace
            if tracing_code:
                tracename = disp.source_filename
                assert tracename is not None
                if tracename not in self.data:
                    self.data[tracename] = set()    # type: ignore[assignment]
                self.cur_file_data = self.data[tracename]
            else:
                self.cur_file_data = None

        self.code_cache[code] = (tracing_code, self.cur_file_data)

        if tracing_code:
            events = sys.monitoring.events
            log(f"set_local_events({code!r})")
            sys.monitoring.set_local_events(
                self.myid,
                code,
                (
                    sys.monitoring.events.LINE |
                    sys.monitoring.events.PY_RETURN |
                    sys.monitoring.events.PY_RESUME |
                    sys.monitoring.events.PY_YIELD
                )
            )

        self.last_line = -code.co_firstlineno

    @panopticon
    def sysmon_py_resume(self, code, instruction_offset: int):
        self.frame_stack.append((self.cur_file_data, self.cur_file_name, self.last_line))
        frame = inspect.currentframe()
        self.last_line = frame.f_lineno

    @panopticon
    def sysmon_py_return(self, code, instruction_offset: int, retval: object):
        if self.cur_file_data is not None:
            cast(Set[TArc], self.cur_file_data).add((self.last_line, -code.co_firstlineno))

        # Leaving this function, pop the filename stack.
        self.cur_file_data, self.cur_file_name, self.last_line = (
            self.frame_stack.pop()
        )

    def sysmon_py_yield(self, code, instruction_offset: int, retval: object):
        ...

    @panopticon
    def sysmon_line(self, code, line_number: int):
        #assert self.cur_file_data is not None
        if self.cur_file_data is not None:
            if self.trace_arcs:
                cast(Set[TArc], self.cur_file_data).add((self.last_line, line_number))
            else:
                cast(Set[TLineNo], self.cur_file_data).add(line_number)
            self.last_line = line_number
        return sys.monitoring.DISABLE

    @panopticon
    def sysmon_branch(self, code, instruction_offset: int, destination_offset: int):
        ...
