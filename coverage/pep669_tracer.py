# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/nedbat/coveragepy/blob/master/NOTICE.txt

"""Raw data collector for coverage.py."""

from __future__ import annotations

import atexit
import dataclasses
import dis
import inspect
import re
import sys
import threading
import traceback

from types import CodeType, FrameType, ModuleType
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, cast

from coverage.types import (
    TArc, TFileDisposition, TLineNo, TTraceData, TTraceFileData, TTraceFn,
    TTracer, TWarnFn,
)

# When running meta-coverage, this file can try to trace itself, which confuses
# everything.  Don't trace ourselves.

THIS_FILE = __file__.rstrip("co")


def logfile():
    with open("/tmp/pan.out", "a") as f:
        yield f

def log(msg):
    for f in logfile():
        print(msg, file=f)

FILENAME_SUBS = [
    (r"/private/var/folders/.*/pytest-of-.*/pytest-\d+/", "/tmp/"),
]

def arg_repr(arg):
    match arg:
        case CodeType():
            filename = arg.co_filename
            for pat, sub in FILENAME_SUBS:
                filename = re.sub(pat, sub, filename)
            arg_repr = f"<name={arg.co_name}, file={filename!r}@{arg.co_firstlineno}>"
        case _:
            arg_repr = repr(arg)
    return arg_repr

def panopticon(*names):
    def _decorator(meth):
        def _wrapped(self, *args, **kwargs):
            assert not kwargs
            try:
                args_reprs = []
                for name, arg in zip(names, args):
                    if name is None:
                        continue
                    args_reprs.append(f"{name}={arg_repr(arg)}")
                log(f"{meth.__name__}({', '.join(args_reprs)})")
                return meth(self, *args)
            except:
                with open("/tmp/pan.out", "a") as f:
                    traceback.print_exception(sys.exception(), file=f)
                    sys.monitoring.set_events(sys.monitoring.COVERAGE_ID, 0)
                raise
        return _wrapped
    return _decorator


@dataclasses.dataclass
class CodeInfo:
    tracing: bool
    file_data: Optional[TTraceFileData]
    byte_to_line: Dict[int, int]


def bytes_to_lines(code):
    b2l = {}
    cur_line = None
    for inst in dis.get_instructions(code):
        if inst.starts_line is not None:
            cur_line = inst.starts_line
        b2l[inst.offset] = cur_line
    log(f"--> bytes_to_lines: {b2l!r}")
    return b2l

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

        self.code_infos: Dict[CodeType, CodeInfo] = {}
        self.stats = {
            "starts": 0,
        }

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
        sys.monitoring.set_events(
            self.myid,
            events.PY_START | events.PY_RETURN | events.PY_RESUME | events.PY_YIELD,
        )
        sys.monitoring.register_callback(self.myid, events.PY_START, self.sysmon_py_start)
        sys.monitoring.register_callback(self.myid, events.PY_RESUME, self.sysmon_py_resume)
        sys.monitoring.register_callback(self.myid, events.PY_RETURN, self.sysmon_py_return)
        sys.monitoring.register_callback(self.myid, events.PY_YIELD, self.sysmon_py_yield)
        # UNWIND is like RETURN/YIELD
        sys.monitoring.register_callback(self.myid, events.LINE, self.sysmon_line)
        sys.monitoring.register_callback(self.myid, events.BRANCH, self.sysmon_branch)
        sys.monitoring.register_callback(self.myid, events.JUMP, self.sysmon_jump)

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
        return self.stats | {
            "codes": len(self.code_infos),
            "codes_tracing": sum(1 for ci in self.code_infos.values() if ci.tracing),
        }

    @panopticon("code", "@")
    def sysmon_py_start(self, code, instruction_offset: int):
        # Entering a new frame.  Decide if we should trace in this file.
        self._activity = True
        self.stats["starts"] += 1

        self.frame_stack.append((self.cur_file_data, self.cur_file_name, self.last_line))

        code_info = self.code_infos.get(code)
        if code_info is not None:
            tracing_code = code_info.tracing
            self.cur_file_data = code_info.file_data
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
                b2l = bytes_to_lines(code)
            else:
                self.cur_file_data = None
                b2l = None

            self.code_infos[code] = CodeInfo(
                tracing=tracing_code,
                file_data=self.cur_file_data,
                byte_to_line=b2l,
            )

            if tracing_code:
                events = sys.monitoring.events
                log(f"set_local_events(code={arg_repr(code)})")
                sys.monitoring.set_local_events(
                    self.myid,
                    code,
                    sys.monitoring.events.LINE |
                    sys.monitoring.events.BRANCH |
                    sys.monitoring.events.JUMP,
                )

        self.last_line = -code.co_firstlineno

    @panopticon("code", "@")
    def sysmon_py_resume(self, code, instruction_offset: int):
        self.frame_stack.append((self.cur_file_data, self.cur_file_name, self.last_line))
        frame = inspect.currentframe()
        self.last_line = frame.f_lineno

    @panopticon("code", "@", None)
    def sysmon_py_return(self, code, instruction_offset: int, retval: object):
        if self.cur_file_data is not None:
            if self.trace_arcs:
                cast(Set[TArc], self.cur_file_data).add((self.last_line, -code.co_firstlineno))

        # Leaving this function, pop the filename stack.
        if self.frame_stack:
            self.cur_file_data, self.cur_file_name, self.last_line =  self.frame_stack.pop()

    def sysmon_py_yield(self, code, instruction_offset: int, retval: object):
        ...

    @panopticon("code", "line")
    def sysmon_line(self, code, line_number: int):
        #assert self.cur_file_data is not None
        if self.cur_file_data is not None:
            if self.trace_arcs:
                cast(Set[TArc], self.cur_file_data).add((self.last_line, line_number))
            else:
                cast(Set[TLineNo], self.cur_file_data).add(line_number)
            self.last_line = line_number
        #return sys.monitoring.DISABLE

    @panopticon("code", "from@", "to@")
    def sysmon_branch(self, code, instruction_offset: int, destination_offset: int):
        ...

    @panopticon("code", "from@", "to@")
    def sysmon_jump(self, code, instruction_offset: int, destination_offset: int):
        ...
