"""NUMA placement for the factor pool and the sweep worker pools.

Why this exists (measured 2026-11-23, ``dev/poolfloor.py`` / ``dev/numamap.py``
on ``c5g7_2d_quarter_core::M16_S2``, 60 threads, ``--opt chol-rs``): one sweep
costs 320-350 ms while the 21 (i, g) jobs sum to 1.57 s serial and are nearly
equal in length (mean 75 ms, max 88 ms), and the shared-memory trip plus the
parent-side reduce is only ~15 ms.  Run one job alone inside a worker and it
takes what it takes in the parent; run all 21 at once and every job takes
2.4x longer (median 75 -> 180 ms).  The work does not get bigger because of
Python — it gets bigger because of where the bytes are.

The box is 4-node (4x Xeon E5-4640).  Left alone, the factor pool lands wherever
the six prefactor threads happened to run — measured 61% node0 / 22% node1 / 9%
node2 / 8% node3 — while the fork pool's 21 workers spread evenly over all four
nodes, so most of the 1.8 GiB a sweep streams is a REMOTE read and node0's
memory controllers serve most of it.  Blanket policies were measured and all
lose: ``numactl --interleave=all`` 443 ms (and the single-job cost rises 73 ->
86 ms because every random probe now crosses all four home agents),
``--preferred=0`` 473 ms, ``--cpunodebind=0`` 447 ms with the in-flight median
at 347 ms (one node cannot carry 21 readers).

What does work is to stop handing all 21 jobs to one pool that spreads them
evenly over four nodes, and instead give each node its own pool and submit each
system to the pool of its ``home(i, g)`` node — ~5 readers per node instead of
21 spread across all of them, and no queue waiting behind another node's job.
Two further mechanisms are available and were measured separately, because they
are NOT equally useful (``dev/sweepms.py``, paired cycles, shared box):

  * ``pool`` — routing only, workers keep the parent's affinity.  279 → 199 ms
    and 297 → 219 ms; this is where the win lives;
  * ``mem``  — first-touch each factor on its home node in the prefactor
    (:func:`set_mem`).  Adds nothing measurable on top of routing, and alone
    (without routing) it is a pessimisation, which is why ``=1`` refuses to run
    half-placed;
  * ``cpu``  — also pin each pool to its node.  As fast as ``pool`` on a quiet
    box, worse on a shared one (221/292 vs 199/219): a pinned worker cannot
    step away from a node a co-tenant is hammering, so somebody else's load
    becomes this sweep's wall time.

Placement cannot change the ORDER of the arithmetic: each worker writes its
reduce into its own (g, i) slot of shared memory and the parent still
accumulates in (g, i) order.  Now that the compiled kernel pins METIS's seed
and serialises its ordering calls (rust/src/metis.rs), that is testable with
exact equality rather than a tolerance, and it holds: ``M4_S2`` gives
1.1849017894828735 and ``M8_S2`` 1.3401628545523656 with ``OPENPSN_NUMA=pool``
and with it unset, identical to the last digit and with identical iteration
counts.  (Until that kernel fix, the same comparison wobbled by ~1e-11 — which
was the ordering being redrawn per process, not anything about placement.)

Everything here is OFF unless ``OPENPSN_NUMA=1``: on a single-node machine, or
one where ``set_mempolicy`` is blocked (seccomp), the solver keeps the behaviour
it has always had.  ``status()`` reports what it actually did.
"""

import ctypes
import os
from pathlib import Path

# __NR_set_mempolicy on x86-64 is 238 (239 is get_mempolicy — passing that one
# is what makes a correct-looking call return EINVAL).  ctypes rather than
# libnuma so the solver keeps no new dependency; the numbers and the mask layout
# are asserted at first use by _try_init's bind/reset round trip, so a wrong
# guess degrades to "placement off" instead of a silently misplaced pool.
_SYS_set_mempolicy = 238
_MPOL_DEFAULT, _MPOL_BIND = 0, 2

_state = {"tried": False, "mem_ok": False, "nodes": [], "err": ""}
_libc = None


def _libc_handle():
    global _libc
    if _libc is None:
        _libc = ctypes.CDLL("libc.so.6", use_errno=True)
    return _libc


def enabled():
    """User opt-in.  Default off, so nothing changes for anyone who does not
    ask for it (and the bit-identical default path stays the reference).

    Placement is three independent mechanisms, worth different amounts on
    different boxes, so each can be asked for alone (and measured alone):

      * ``mem``  — ``set_mempolicy`` first-touch of each (i, g) factor on one
        node, in the prefactor;
      * ``pool`` — one worker pool PER NODE, with the sweep routed to the data's
        home node.  The worker keeps the parent's affinity, so the scheduler can
        still move it off a node a co-tenant is hammering;
      * ``cpu``  — as ``pool``, plus each pool is forked while the parent is
        affinity-bound to its node, so the workers cannot leave it.  Usually the
        fastest, and the most fragile: a pinned worker that cannot migrate
        turns somebody else's load on that node into this sweep's wall time
        (measured: one leg of eight at 357 ms against 205 ms on the same
        settings minutes earlier).

    ``OPENPSN_NUMA=1`` (or ``all``) asks for mem + pool and gets NEITHER if the
    kernel refuses ``set_mempolicy``: routing jobs to a node the bytes never
    landed on is worse than not routing at all, so the combined mode is
    all-or-nothing.  On a kernel that refuses it, ``=pool`` is the part that
    still applies.  ``=cpu`` adds pinning and is meant for a DEDICATED machine:
    measured on a shared one it is the worst of the three (292 ms against
    219 ms for the same run unpinned), because a worker that cannot migrate
    waits out whatever a co-tenant is doing on its node.
    """
    return bool(modes())


_MEM, _POOL, _CPU = "mem", "pool", "cpu"


def modes():
    """Modes requested by the environment (before probing what works)."""
    v = os.environ.get("OPENPSN_NUMA", "").strip().lower()
    if v in ("1", "true", "yes", "all"):
        return (_MEM, _POOL)
    if v in ("mem", "memory"):
        return (_MEM,)
    if v == _POOL:
        return (_POOL,)
    if v == _CPU:
        return (_POOL, _CPU)          # pinning only exists per pool
    return ()


def nodes():
    """[(node_id, frozenset(cpus))] from sysfs, empty if unreadable/single-node.

    A single-node box returns [] on purpose: there is nothing to place, and the
    caller then keeps the plain path instead of paying for four idle pools.
    """
    out = []
    try:
        for d in sorted(Path("/sys/devices/system/node").glob("node[0-9]*")):
            txt = (d / "cpulist").read_text().strip()
            cpus = set()
            for part in txt.split(","):
                if "-" in part:
                    a, b = part.split("-")
                    cpus.update(range(int(a), int(b) + 1))
                elif part != "":
                    cpus.add(int(part))
            if cpus:
                out.append((int(d.name[4:]), frozenset(cpus)))
    except OSError as exc:                     # no sysfs, container, ...
        _state["err"] = f"sysfs: {exc}"
        return []
    return out if len(out) > 1 else []


def _try_init():
    _state["tried"] = True
    _state["nodes"] = nodes()
    if not _state["nodes"]:
        return
    # Probe on the calling thread with a bind+reset round trip: if the kernel
    # refuses (seccomp in some containers), fall back for the whole run rather
    # than failing deep inside the factorisation.
    if set_mem(_state["nodes"][0][0]):
        set_mem(None)
        _state["mem_ok"] = True
    else:
        _state["err"] = _state.get("err") or "set_mempolicy refused"


def active():
    """Modes IN FORCE: requested, and possible on this machine.

    ``cpu`` needs only a readable sysfs topology; ``mem`` additionally needs a
    working ``set_mempolicy``; and a mode that asks for both (``=1``) collapses
    to nothing rather than to ``cpu`` alone, because routing to a home node the
    factor was never placed on is a pessimisation, not a partial win.
    """
    if not _state["tried"]:
        _try_init()
    want = modes()
    if not want or len(_state["nodes"]) < 2:
        return ()
    if _MEM in want and not _state["mem_ok"] and len(want) > 1:
        return ()                     # =1 is all-or-nothing, see modes()
    out = [m for m in want if m != _MEM or _state["mem_ok"]]
    if _CPU in out and _POOL not in out:
        out.append(_POOL)
    return tuple(out)


def status():
    if not _state["tried"]:
        _try_init()
    return {"requested": list(modes()), "active": list(active()),
            "nodes": [n for n, _ in _state["nodes"]],
            "jobs_per_node": _state.get("split", {}),
            "error": _state["err"]}


def mem_home():
    """Are factors being first-touched on a home node?  (Routing that assumes it
    must only be trusted when this is True — ``active()`` keeps the two halves
    consistent, this is the check the sweep pool uses.)"""
    return _MEM in active()


def route_pools():
    """Should the sweep run one pool per node and route each job to its home?"""
    a = active()
    return _POOL in a or _CPU in a


def cpu_pools():
    """Should those pools also be affinity-bound to their node (no migration)?"""
    return _CPU in active()


def width():
    """How many nodes to place across: >1 only when asked AND working.

    Callers use this instead of reading the node list directly so that a refused
    ``set_mempolicy``, a single-node box or an unset environment variable all
    collapse to the plain path (0), which is what the solver had before this
    module existed.
    """
    return len(_state["nodes"]) if active() else 0


def node_cpus():
    """[(node, cpus)] for the placement decisions; [] when placement is off."""
    return _state["nodes"] if width() > 1 else []


def home(i, g, ng, nn=None):
    """Node a system's factor belongs on.  Round-robin over the (i, g) index so
    every node gets the SAME MIX of polar directions and groups (the per-system
    cost is not equal across i, and a block-per-node split would hand one node
    all the expensive ones)."""
    if nn is None:
        nn = len(_state["nodes"]) or 1
    return (i * ng + g) % nn if nn > 1 else 0


def set_mem(node):
    """Pin THIS thread's future allocations to ``node`` (None = back to default).

    Per-thread on Linux, which is what makes it usable from the prefactor's
    ThreadPoolExecutor: two threads factorising two systems simultaneously get
    two different homes.  MPOL_BIND keeps the mask to that one node; if it is
    ever full the kernel falls back to the local node rather than failing.
    """
    if not _state["nodes"]:
        return False
    if node is None:      # MPOL_DEFAULT: the kernel demands NULL and 0
        rc = _libc_handle().syscall(ctypes.c_long(_SYS_set_mempolicy),
                                    ctypes.c_long(_MPOL_DEFAULT),
                                    ctypes.c_void_p(0), ctypes.c_ulong(0))
    else:
        # maxnode is the node-id RANGE (>= 2), the mask is an unsigned long[]
        # with 64 ids per word — both are checked by the kernel, so use the
        # round 64 rather than the highest id seen.
        maxnode = 64
        mask = (ctypes.c_ulong * 1)()
        mask[node // 64] = ctypes.c_ulong(1 << (node % 64))
        rc = _libc_handle().syscall(ctypes.c_long(_SYS_set_mempolicy),
                                    ctypes.c_long(_MPOL_BIND), mask,
                                    ctypes.c_ulong(maxnode))
    if rc != 0:
        _state["err"] = f"set_mempolicy: {os.strerror(ctypes.get_errno())}"
        return False
    return True


def set_cpu(cpus):
    """Affinity for the calling thread (and therefore for anything it forks)."""
    try:
        os.sched_setaffinity(0, cpus)
        return True
    except OSError as exc:
        _state["err"] = f"sched_setaffinity: {exc}"
        return False
