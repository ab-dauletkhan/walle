"""Thread decoupler for moonshine_voice MicTranscriber listener callbacks.

MicTranscriber fires on_line_started / on_line_text_changed / on_line_completed
synchronously on its internal audio thread. Heavy listener work there (embedding
similarity, serial I/O, SSH prints, tts.drain) stalls the audio drain and the
PortAudio ring buffer overflows. This module provides a ListenerDispatcher that
is the only listener registered with MicTranscriber; it enqueues events onto a
bounded deque and a dedicated consumer thread runs the real listener methods.

Partial (on_line_text_changed) events may be dropped under back-pressure; starts
and completions are preserved. The next delivered event carries
`dispatcher_drops_since_last` so listeners can reset any "consecutive partial"
counters.

A stop-intent fast lane lets the audio thread fire a synchronous stop callback
before enqueue so barge-in latency is not bounded by a busy consumer.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

try:  # pragma: no cover - exercised on voice-less hosts
    from moonshine_voice import TranscriptEventListener  # type: ignore
except ImportError:

    class TranscriptEventListener:  # type: ignore[no-redef]
        """Stub base used when moonshine_voice is unavailable."""


_LINE_STARTED = "line_started"
_LINE_TEXT_CHANGED = "line_text_changed"
_LINE_COMPLETED = "line_completed"

_SLOW_LISTENER_MS = 120.0  # per-call threshold for a WARNING log
_STATS_INTERVAL_S = 30.0


@dataclass
class _Event:
    kind: str
    payload: object
    enqueued_ts: float


@dataclass
class _MethodStats:
    count: int = 0
    sum_s: float = 0.0
    max_s: float = 0.0
    samples: deque = field(default_factory=lambda: deque(maxlen=200))


class DispatcherStats:
    """Thread-safe counters + per-(listener, kind) timings.

    Updated from the consumer thread and read by the 30 s summary timer.
    A single lock guards the whole struct; contention is low (one writer).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.queue_depth_hwm = 0
        self.dropped_partial = 0
        self.forced_drop = 0
        # Total events received on the audio thread. Used by the router
        # heartbeat to distinguish "mic is silent" (counter stays 0) from
        # "mic is alive but gated" (counter climbs).
        self.audio_thread_events = 0
        self._methods: dict[tuple[str, str], _MethodStats] = {}

    def observe_depth(self, depth: int) -> None:
        with self._lock:
            if depth > self.queue_depth_hwm:
                self.queue_depth_hwm = depth

    def record(self, listener: str, kind: str, dur_s: float) -> None:
        with self._lock:
            key = (listener, kind)
            stats = self._methods.get(key)
            if stats is None:
                stats = _MethodStats()
                self._methods[key] = stats
            stats.count += 1
            stats.sum_s += dur_s
            if dur_s > stats.max_s:
                stats.max_s = dur_s
            stats.samples.append(dur_s)

    def snapshot_and_reset_window(self) -> str:
        """Format a single-line summary and reset the rolling samples."""
        with self._lock:
            parts = [
                f"qdepth_hwm={self.queue_depth_hwm}",
                f"dropped_partial={self.dropped_partial}",
                f"forced_drop={self.forced_drop}",
            ]
            for (listener, kind), stats in sorted(self._methods.items()):
                if not stats.samples:
                    continue
                s = sorted(stats.samples)
                p50 = s[len(s) // 2] * 1000.0
                p95 = s[int(len(s) * 0.95)] * 1000.0 if len(s) >= 2 else p50
                mx = stats.max_s * 1000.0
                parts.append(
                    f"{listener}.{kind}[n={stats.count},"
                    f"p50={p50:.1f}ms,p95={p95:.1f}ms,max={mx:.1f}ms]"
                )
                stats.samples.clear()
                stats.max_s = 0.0
            self.queue_depth_hwm = 0
            return " ".join(parts)


class ListenerDispatcher(TranscriptEventListener):
    """Bounded-queue decoupler between MicTranscriber and real listeners.

    Register this as the sole listener on MicTranscriber; pass the real
    listeners via the constructor. The audio thread's path is O(enqueue)
    except in the drop case, which is still lock-local and O(queue_size).
    """

    def __init__(
        self,
        listeners: list,
        queue_size: int = 32,
        stop_intent_fn: Optional[Callable[[str], bool]] = None,
        stop_active_fn: Optional[Callable[[], bool]] = None,
        on_stop_intent: Optional[Callable[[], None]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__()
        self._listeners = list(listeners)
        self._queue_size = queue_size
        self._stop_intent_fn = stop_intent_fn
        self._stop_active_fn = stop_active_fn
        self._on_stop_intent = on_stop_intent
        self._log = logger or logging.getLogger("walle.voice.dispatcher")

        self._deque: deque[_Event] = deque()
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        # Drops are attached to the NEXT event the consumer delivers, not
        # the event currently being enqueued. Rationale: if we drop partial
        # #3 of [1,2,3,4,5] the consumer should see 4 tagged with drops=1
        # so "consecutive-partial" state machines can reset on the right
        # event. Tagging the incoming event at append time instead would
        # attach the drop to a future event that isn't in the sequence yet.
        self._pending_drops_for_next_delivered = 0
        self._shutdown = threading.Event()

        self.stats = DispatcherStats()

        self._consumer = threading.Thread(
            target=self._run, name="dispatch-consumer", daemon=True
        )
        self._consumer.start()

        self._stats_timer: Optional[threading.Timer] = None
        self._schedule_stats_log()

    # -- TranscriptEventListener callbacks (run on moonshine's audio thread) --

    def _note_audio_event(self, kind: str) -> None:
        self.stats.audio_thread_events += 1
        if self.stats.audio_thread_events == 1:
            # First event ever — confirms Moonshine's audio thread is alive.
            # Logged once; after that the heartbeat shows the counter.
            self._log.info(
                "dispatcher: first audio-thread event received (kind=%s)",
                kind,
            )

    def on_line_started(self, event):
        self._note_audio_event(_LINE_STARTED)
        self._enqueue(_LINE_STARTED, event)

    def on_line_text_changed(self, event):
        # Stop-intent fast lane: barge-in latency must not be bounded by a
        # busy consumer (e.g. a 100 ms embedding call). Match + fire stop
        # synchronously here; the event still goes through the queue so
        # the router's state machine sees it normally.
        if self._stop_intent_fn is not None and self._stop_active_fn is not None:
            try:
                text = getattr(getattr(event, "line", None), "text", "") or ""
                if text and self._stop_active_fn() and self._stop_intent_fn(text):
                    if self._on_stop_intent is not None:
                        self._on_stop_intent()
            except Exception:  # never let the fast lane kill audio
                self._log.exception("stop-intent fast lane failed")
        self._note_audio_event(_LINE_TEXT_CHANGED)
        self._enqueue(_LINE_TEXT_CHANGED, event)

    def on_line_completed(self, event):
        self._note_audio_event(_LINE_COMPLETED)
        self._enqueue(_LINE_COMPLETED, event)

    # -- Producer side --

    def _enqueue(self, kind: str, payload: object) -> None:
        now = time.monotonic()
        with self._lock:
            if len(self._deque) >= self._queue_size:
                # Make room by dropping the OLDEST partial in the queue.
                dropped_here = self._drop_oldest_partial_locked()
                if dropped_here == 0:
                    # Pathological: queue full of starts/completions.
                    # Drop the incoming event (never block the audio thread)
                    # and escalate — losing a line_started or line_completed
                    # is meaningful, not a routine stat.
                    self.stats.forced_drop += 1
                    self._log.error(
                        "dispatcher forced drop: kind=%s qsize=%d "
                        "(queue saturated with non-droppable events)",
                        kind,
                        len(self._deque),
                    )
                    return
                self.stats.dropped_partial += 1
                self._pending_drops_for_next_delivered += 1

            self._deque.append(_Event(kind, payload, now))
            depth = len(self._deque)
            self._cond.notify()
        self.stats.observe_depth(depth)

    def _drop_oldest_partial_locked(self) -> int:
        """Remove the oldest on_line_text_changed from the deque. Caller
        must hold self._lock. Returns 1 if a partial was dropped, 0 if
        the deque contained no partials.
        """
        for i, ev in enumerate(self._deque):
            if ev.kind == _LINE_TEXT_CHANGED:
                del self._deque[i]
                return 1
        return 0

    # -- Consumer side --

    def _run(self) -> None:
        while True:
            with self._cond:
                while not self._deque and not self._shutdown.is_set():
                    self._cond.wait()
                if self._shutdown.is_set() and not self._deque:
                    return
                event = self._deque.popleft()
                drops_since_last = self._pending_drops_for_next_delivered
                self._pending_drops_for_next_delivered = 0

            # Attach drop count so SpeechRouter can reset _partial_stop_hits.
            # Setting an attribute on the library's event object is
            # non-invasive; existing code never references this name.
            try:
                setattr(event.payload, "dispatcher_drops_since_last", drops_since_last)
            except Exception:
                pass  # payload may be an odd type on some versions

            for listener in self._listeners:
                method = getattr(listener, f"on_{event.kind}", None)
                if method is None:
                    continue
                listener_name = type(listener).__name__
                t0 = time.perf_counter()
                try:
                    method(event.payload)
                except Exception:
                    self._log.exception(
                        "listener %s.on_%s raised", listener_name, event.kind
                    )
                dur = time.perf_counter() - t0
                self.stats.record(listener_name, event.kind, dur)
                if dur * 1000.0 >= _SLOW_LISTENER_MS:
                    self._log.warning(
                        "slow listener: %s.on_%s took %.1f ms",
                        listener_name,
                        event.kind,
                        dur * 1000.0,
                    )

    # -- Stats timer --

    def _schedule_stats_log(self) -> None:
        if self._shutdown.is_set():
            return
        self._stats_timer = threading.Timer(_STATS_INTERVAL_S, self._log_stats)
        self._stats_timer.daemon = True
        self._stats_timer.start()

    def _log_stats(self) -> None:
        if self._shutdown.is_set():
            return
        summary = self.stats.snapshot_and_reset_window()
        self._log.info("dispatcher stats: %s", summary)
        self._schedule_stats_log()

    # -- Shutdown --

    def stop(self, drain_timeout: float = 0.5) -> None:
        """Stop the consumer. Idempotent.

        Waits up to `drain_timeout` seconds for the consumer to finish
        any in-flight listener call, then returns. Any events still on
        the queue past the timeout are discarded when the daemon thread
        exits — better to lose queued text than crash on torn-down
        dependencies during shutdown.
        """
        if self._shutdown.is_set():
            return
        self._shutdown.set()
        with self._cond:
            self._cond.notify_all()
        if self._stats_timer is not None:
            self._stats_timer.cancel()
        self._consumer.join(drain_timeout)
        # Final stats line regardless of whether we drained cleanly.
        self._log.info(
            "dispatcher final: %s", self.stats.snapshot_and_reset_window()
        )
