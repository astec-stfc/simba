.. _ring-studies:

Ring Studies and the ``tracking`` Block
=======================================

:ref:`Loading a lattice <loading-a-lattice>` describes the ``files:`` block, which
says *what* each line is and which code runs it. A ring needs one more thing: how
the line is to be *tracked*. That is the ``tracking:`` block, and it sits inside a
``files:`` entry alongside ``code``, ``input`` and ``output``:

.. code-block:: yaml

    files:
      RING:
        code: elegant
        tracking:
          turns: 1000
          periodic: true
          radiation: quantum
          write_turns: false

Everything in this block is a property of the *study*, not of the machine. The
lattice is `LAURA <https://github.com/astec-stfc/laura/>`_'s and is the same
whether it is tracked once or a million times; how many times, from what optics,
with what physics switched on, and what is written out belongs to SIMBA.
That line is the reason these settings live here rather than in the layout, and it
is drawn again, more finely, in :ref:`device-programs`.

.. note::
   Not every code can honour every setting, and a code that cannot **says so** and
   carries on rather than quietly doing something else. The warnings name the codes
   that can, read off the codes' own ``supports_*`` flags
   (:meth:`~simba.Framework_objects.frameworkLattice.codes_that_can`) so the list
   cannot fall behind them. See :ref:`ring-capabilities`, and
   :ref:`simba-warnings` for silencing them by kind.

.. note::
   **Space charge and CSR are the line's, and LSC is off unless asked for.**
   ``lsc_enable`` defaults to ``false`` in LAURA and in every SIMBA code.
   A line that wants longitudinal space charge says ``lsc_enable: true`` in its
   ``files:`` entry; a linac deck that relied on the old default needs to.
   ``csr_enable`` is unchanged.

.. _tracking-settings:

The Settings
------------

turns
^^^^^^^^^

How many times the line is tracked, the beam leaving the last element re-entering
the first. Defaults to ``1``.

.. code-block:: yaml

    tracking: {turns: 1000}

A ring spanning several sections is written as one section naming them — section
orders nest — so the unit a turn applies to is the whole line.

A line that is tracked for more than one turn, or asked for the periodic solution,
is also checked for *closure*: if the first element's entrance and the last
element's exit are more than a tolerance apart, that raises an error. The
legitimate exception is a **superperiod**, one sector of an N-fold symmetric ring,
and the check recognises it from the net bend angle and names the
``nsuperperiods`` that would make it close.

nsuperperiods
^^^^^^^^^^^^^^^^^

How many times the line is traversed per turn. One sector of an N-fold symmetric
ring is written once and tracked N times, rather than written out N times:

.. code-block:: yaml

    tracking: {turns: 1000, nsuperperiods: 4}

**A turn stays a turn.** Those settings track 4000 passes through the sector, and
the beam files, the turn suffixes, the device-program turn axis and everything
else counted per turn still count 1000 of them. A screen gives one file per turn,
not four.

Two checks come with it. The closure test above is replaced by the question that
is actually meaningful for a sector — whether *N* of them bend through a whole
number of turns — which catches the count being wrong, otherwise invisible:
tracking six sectors of a four-fold ring runs, converges, and describes a machine
that does not exist. And a code that cannot repeat the line warns **loudly**,
because unlike every other unsupported setting, ignoring this one does not give
a coarser run, it tracks one Nth of the intended ring.

The mechanism differs and the semantics do not. Ocelot's ``track_nturns`` takes
the count natively and its bunch path loops the sector inside simba's turn loop.
MAD-X multiplies it into the single ``RUN, TURNS=N`` described in
:ref:`native-turns` below, so a superperiod is a pass there too and turn ``t``
is MAD-X's turn ``t × nsuperperiods``. Xtrack has no notion
of a sector at all, so ``num_turns`` is simply multiplied and everything counted
per turn is converted back — ``at_turn`` counts passes and is divided down, and
a monitor records a pass, so the turn boundaries are taken every Nth sample.

.. note::
   With ``nsuperperiods`` set, ``ring_parameters()`` describes the **sector**,
   not the ring. A sector's phase advance can exceed :math:`2\pi` and the
   reported tune is fractional, so N times the sector tune is not the ring tune;
   reconstructing one needs the integer part from somewhere else.

periodic
^^^^^^^^^^^^

Whether to ask the code for the closed (matched) optics solution rather than
propagating the incoming beam's Twiss.

Unlike the others this is not *primarily* a tracking setting — whether the
reference orbit closes is a fact about the lattice, and LAURA already records it as
the section's ``geometry``. The ``tracking`` block overrides that either way, which
is what an injection-mismatch study on a real ring needs:

.. code-block:: yaml

    tracking: {turns: 1000, periodic: false}

radiation
^^^^^^^^^^^^^

The synchrotron-radiation model: ``mean`` gives damping and the energy loss,
``quantum`` adds the excitation, and only with both does an equilibrium emittance
exist.

.. code-block:: yaml

    tracking: {turns: 100000, radiation: quantum}

Omitting it is not the same as switching it off. With no setting, every code is
left on its own default — and those defaults are *not the same*: elegant and Bmad
radiate with no asking, Xsuite and Ocelot do not. A long lepton ring tracked with
radiation off warns, because without damping and excitation the emittance, the energy
spread and bunch length at the end are the ones the run started with rather than
the ring's.

write_turns
^^^^^^^^^^^^^^^

Whether a multi-turn run writes a beam file per turn. **Off by default**:

.. code-block:: yaml

    tracking: {turns: 1000, write_turns: true}

With it off, a multi-turn run writes what a single-turn run writes — one file per
screen, marker and BPM, and one at the end of the line, holding the last turn,
under the unsuffixed name — so a ring run looks like any other run to everything
downstream. With it on you get ``M1-t1`` … ``M1-tN`` *and* the unsuffixed file,
which is the one the next section reads by name; see :ref:`which-turn` for how
to tell them apart from the inside.

Every ring code records at the same places: screens, markers, BPMs and the end
of the line. The start element's unsuffixed file is the input beam, and is never
written over. Under ``nsuperperiods``, turn ``k`` is the beam on the last pass of that
turn. The s written is the element's position in the machine, measured from the
lattice's own entrance, and never the incoming beam's ``s``.

single_particle
^^^^^^^^^^^^^^^^^^^

Track 13 probe particles instead of the whole bunch and carry the distribution
through the linear map they measure.

.. code-block:: yaml

    tracking: {single_particle: true}

A linear reconstruction, and a large speed-up where the code's cost is per
particle. Currently MAD-X only.

dynamic_aperture
^^^^^^^^^^^^^^^^^^^^

The grid for a dynamic-aperture scan or a frequency map:

.. code-block:: yaml

    tracking:
      turns: 1000
      dynamic_aperture: {nx: 20, ny: 10, x_max: 0.02, y_max: 0.01}

Both axes start one step off zero rather than at it.

.. _device-programs:

programs — a Strength Over Turn Number
------------------------------------------

An injection kicker or an extraction septum is not a static element: its strength
is a *program over turn number*, zero for the first N turns, up for one, and down
again.

The authoring of such a device is **split between the two repositories**, along the
line the tracking codes themselves draw. The pulse *shape* — rise and fall time,
flat-top length, whether the device is a single-turn kicker or a slow bumper — is
hardware, is the same in every study, and lives in LAURA as a waveform on the
element. *When it fires and at what amplitude* is the study, and lives here:

.. code-block:: yaml

    files:
      RING:
        code: elegant
        tracking:
          turns: 10
          programs:
            - element: KICK1
              turns:  [1, 4, 5]
              values: [0.0, 1.0e-3, 0.0]
              interpolation: hold

That is an element held at zero, kicked by 1 mrad on turn 4 alone, and back to zero
afterwards.

Three conventions
^^^^^^^^^^^^^^^^^

Each of these is stated explicitly rather than inherited, because the codes do not
agree on any of them and a wrong choice gives a run that tracks perfectly and
models the wrong machine.

**Turns are 1-based.** Turn 1 is the first turn tracked, matching the ``_turn``
suffixes on the beam files. elegant counts passes from 0 and Xsuite's ``t_turn_s``
is :math:`n \cdot T_{rev}` for 0-based :math:`n`, so each backend converts.

**Values are the element's strength as the lattice states it** — a deflection angle
in radians for a steering element, positive toward positive *x* (or *y*), not a
fraction of anything. Each backend converts into its own code's attribute: a
corrector's angle is Ocelot's ``angle``, MAD-X's ``kick`` and elegant's ``ANGLE``
unchanged, but Xtrack's ``knl[0]`` with the **sign reversed**, because a positive
normal multipole deflects the other way there.

**The default rule is** ``hold``, **and that is no code's default.** Every tracking
code joins programmed samples with straight lines. Both of these devices are step
devices, and with the knots above, linear interpolation leaks a third of the kick
on turn 2 and two thirds on turn 3 — before the kicker has fired at all. ``hold``,
``linear`` and ``spline`` are all available; ``hold`` is what a step device wants,
so it is the default and is said out loud here.

Outside the listed turns the value is **clamped** to the first or last knot, which
is what every code's own interpolation does too. A pulse that has to come back down
therefore says so, with a final knot at zero; a slow bumper that stays on simply
stops listing knots. A program whose last value is non-zero while the run continues
past it warns, as does a run that ends part-way through a program.

An optional ``parameter:`` names the code-native attribute to set directly. That
turns off both the plane lookup and the sign conversion, and is the escape hatch
for an element simba has no default for. Without it, an element with both planes
(a MAD-X ``KICKER``, an Xsuite multipole) is set in its own plane, from its LAURA
``hardware_type``, in every code
(:meth:`~simba.Framework_objects.frameworkLattice.program_attribute`). A program
naming an element the line does not have warns in every code too.

How each code does it
^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 12 88

   * - Code
     - Mechanism
   * - Xsuite
     - Natively, with no Python loop: the attribute is bound to a
       ``FunctionPieceWiseLinear`` of ``t_turn_s`` and varies inside a single
       ``line.track(num_turns=N)`` call.
   * - elegant
     - ``&alter_elements`` setting ``FIRE_ON_PASS`` and a ``WAVEFORM`` SDDS
       sidecar on a ``BUMPER``. **Not** ``&ramp_elements``, which cannot express
       a pulse. elegant is the only code here whose answer depends on the element
       *type* — a ``BUMPER``/``MBUMPER`` is the only thing that carries a
       ``WAVEFORM``, so a programmed device should be modelled as an AC dipole,
       which LAURA writes as one.
   * - Ocelot
     - The attribute set on the sequence element between turns of simba's loop.
   * - MAD-X
     - The attribute tied to a MAD-X variable (``NAME, ATTR:=var``) as each
       segment is defined, before ``MAKETHIN``, and the variable set between
       turns. The variable can be
       set before its segment exists, which on turn 1 it does not. A
       **multipole cannot be programmed in MAD-X**: it kicks by
       :math:`-(K_0L - \mathrm{ANGLE})`, and ``ANGLE`` defaults to ``KNL[0]``.
       A program on one warns and is not applied; model the device as a kicker.
   * - Bmad
     - ``set element`` inside the Tao turn loops, which are the only place Bmad
       counts turns here.

.. _energy-ramp:

ramp — the Reference Momentum Over Turn Number
------------------------------------------------

A booster ramp, stated once and read the same way by every code that can track
one:

.. code-block:: yaml

    files:
      RING:
        code: xsuite
        tracking:
          turns: 2000
          ramp:
            turns: [1, 1000, 2000]
            kinetic_energy: [160.0e6, 2.0e9, 2.0e9]

``momentum`` (``p0c``, eV) may be given instead of ``kinetic_energy`` (eV), but
not both. Turns are 1-based, as for ``programs``. The default interpolation is
``linear``, since a ramp is smooth where a kicker is not; ``hold`` and ``spline``
are also available.

The model
^^^^^^^^^

Every backend implements this and nothing else:

* the ramp sets the **reference** momentum at the start of each turn and holds
  it for the whole turn, including across superperiods;
* changing the reference **changes no particle**. Absolute energy, absolute
  transverse momentum and arrival time are kept; only coordinates measured
  from the reference move;
* magnet strengths are **normalised**, so they stay put and the fields follow
  the reference;
* **the RF does the accelerating.** A bunch whose cavities are phased for a
  stationary bucket slides to the synchronous phase on its own. With too little
  voltage it falls off the ramp.

Where a code needs time rather than turn number, turn :math:`n` starts at

.. math::

   t_1 = 0, \qquad
   t_{n+1} = t_n + \frac{C}{c\,\tfrac12\left(\beta_0(n) + \beta_0(n+1)\right)}

with :math:`C` the length of one pass. The mid-point is Xsuite's own convention.
Written with one knot per pass, it puts Xsuite on integer turns exactly. A
:ref:`device program <device-programs>` in a ramped run is timed on the same
clock.

A ramp warns, and is otherwise tracked as written, when:

* the code cannot ramp (Bmad -- to be implemented);
* the run is one turn, or stops part-way up the ramp;
* the beam does not start on it (more than 0.1% off);
* there is no RF cavity, or the cavities' total voltage is less than the
  largest energy step a turn needs.

How each code does it
^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 12 88

   * - Code
     - Mechanism
   * - Xsuite
     - Natively: an ``xt.EnergyProgram`` with one knot per pass on the clock
       above. Xtrack re-references the particles every turn
       (``update_p0c_and_energy_deviations``). simba's
       ``ReferenceEnergyIncrease`` before each cavity is left out, because
       the program owns the reference.
   * - elegant
     - Natively: a ``RAMPP`` first in the beamline, reading a ``WAVEFORM``
       sidecar. ``RAMPP`` samples its waveform at the **bunch's mean arrival
       time** rather than at any reference clock.
       A staircase is used on that axis, flat for a quarter
       turn either side of each turn's start. ``run_setup`` is written with
       ``always_change_p0 = 0``: re-centring ``p_central`` on the beam after
       every element undoes the ramp entirely.
   * - Ocelot
     - Between turns of SIMBA's loop: the energy is moved and each particle's
       ``p``, ``x'`` and ``y'`` are re-expressed so that its absolute momentum
       is kept. Ocelot's own ``LatticeEnergyProfile`` is not used; it has the
       electron mass written in and leaves ``x'`` and ``y'`` alone.
   * - MAD-X
     - Between turns of SIMBA's loop: the ring's reference ``p0c`` and time are
       carried from turn to turn rather than taken from the bunch.
   * - Bmad
     - Not supported yet; its tracking path here is single-pass.

``unit_tests/test_energy_ramp.py`` tracks one ramped ring through all four
codes. Over a 2% ramp with no RF, the ramp moves the orbit by about 1.2e-4 m.
Xsuite and Ocelot agree with elegant to 1e-7 m and 3e-6 m. MAD-X, whose
lenses are thin, agrees to 2e-5 m.

.. _rf-mode:

rf — How the RF Keeps Time
--------------------------

Left alone, the codes do not agree on what a cavity is. elegant's ``RFCA`` is a
free-running oscillator on absolute time. MAD-X, Xsuite and Ocelot re-phase
their cavities to the reference particle on every pass. On a flat ring with
the cavity on a harmonic you cannot tell the two apart. Under a ramp, where
:math:`\beta_0` changes, they part. They also part on a flat ring whose cavity
is off the harmonic: elegant's beam slips and the others' never does.

So ``rf`` says what the RF does, and every code is made to do it:

.. code-block:: yaml

    tracking: {turns: 2000, ramp: {...}, rf: follow}

``follow`` (the default)
    The frequency follows the beam, :math:`f(n) = f\,\beta_0(n)/\beta_0(1)`,
    as a booster's low-level RF does. The ``frequency`` given is the one at
    turn 1.
``fixed``
    A free-running oscillator at the ``frequency`` given.

It is a ``tracking`` setting rather than part of ``ramp``, because it matters
on a flat ring too, for a cavity off the harmonic.

Each code is moved, pass by pass, from what it does natively to what is
asked. The correction is the phase slip the mode asks for less the code's
own. The reference reaches a cavity at
:math:`T_j(s) = \sum_{k<j} C/(\beta_k c) + s/(\beta_j c)`.

.. list-table::
   :header-rows: 1
   :widths: 12 18 70

   * - Code
     - Natively
     - Moved by
   * - elegant
     - fixed
     - ``&modulate_elements`` on each cavity's ``PHASE``, from an SDDS table
       in the bunch's absolute time, as for ``RAMPP``. ``RAMPRF`` has a
       different phase convention, and ``LOCK_PHASE`` locks to the bunch
       centroid, which is not ``follow``.
   * - Xsuite
     - synchronous
     - each cavity's ``lag`` bound to a staircase in ``t_turn_s``.
   * - MAD-X
     - synchronous
     - each cavity's ``LAG`` between passes of simba's loop. A run whose RF
       has to move cannot use native turns.
   * - Ocelot
     - synchronous
     - each cavity's ``phi`` between passes. Its sign is the opposite of the
       others', because Ocelot's phase is cosine-based.

.. note::
   **Ocelot's cavity moves the reference.** Its cavity map always adds
   :math:`V\cos\phi` to the reference energy. That is right for a linac, but
   it means a ring's beam never slips. On a ring, SIMBA puts the reference
   back at each cavity's exit
   (:class:`~simba.Codes.Ocelot.fixedreference.FixedReference`), keeping
   each particle's absolute momentum. Ocelot's periodic optics with a cavity
   in the ring also needed the reference energy passed in.

.. _ring-capabilities:

What Each Code Can Do
---------------------

A code that cannot honour a setting warns and carries on.

.. list-table::
   :header-rows: 1
   :widths: 15 8 8 10 8 8 10 10 11 10

   * - Code
     - turns
     - periodic
     - radiation
     - programs
     - ramp
     - dynamic aperture
     - frequency map
     - single particle
     - superperiods
   * - elegant
     - yes
     - yes
     - yes (on by default)
     - yes
     - yes
     - yes
     - yes
     - no
     - no
   * - Xsuite
     - yes
     - yes
     - yes
     - yes
     - yes
     - yes
     - yes
     - no
     - yes
   * - Ocelot
     - yes
     - yes
     - yes
     - yes
     - yes
     - yes
     - yes
     - no
     - yes
   * - MAD-X
     - yes
     - yes
     - no
     - yes
     - yes
     - yes
     - yes
     - yes
     - yes
   * - Bmad
     - ring studies only
     - yes
     - yes (on by default)
     - yes
     - no
     - yes
     - yes
     - no
     - no

ASTRA, GPT, Cheetah, OPAL, Wake-T and CSRTrack track a line once and have none of these.

.. _native-turns:

Who Counts the Turns
--------------------

A turn count can be handed to the code or kept in Python, and the difference
is not small. Elegant (``n_passes``) and Xsuite (``line.track(num_turns=…)``)
have always taken the count natively. MAD-X does wherever it can.

When the loop is still used
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each of these needs Python between one turn and the next:

* **more than one segment** — a boundary re-references the beam momentum. With
  the change above this means a linac, which does not have turns anyway;
* **device programs** — :ref:`a program <device-programs>` varies an element per
  turn, which is a MAD-X statement between turns;
* **an energy ramp** — :ref:`the reference momentum <energy-ramp>` changes at
  the start of every turn, and MAD-X has no way to change it inside a ``RUN``;
* **single-particle mode**, which is not one tracking run but a map built by
  finite differences;
* ``native_turns: false``, the explicit override::

    files:
      RING:
        code: madx
        tracking: {turns: 1000, native_turns: false}

The two paths are the same tracking, and ``unit_tests/test_madx_native_turns.py``
holds them to it turn by turn. The override is what lets that test run both.

.. note::
   Ocelot and Bmad still count turns in Python, and neither has a native turn
   count to switch to.

   **Bmad.** Tao has no multi-turn beam tracking: ``beam_init`` carries no turn
   count, ``tao.track_beam`` is one pass, and the only turn counts in Tao are
   the dynamic-aperture scan's (which SIMBA already uses) and a
   ``multi_turn_orbit`` plot curve, which tracks a single particle rather than a
   beam. For general multi-turn Bmad tracking the tool is ``long_term_tracking``
   in ``bsim`` — ``ltt%n_turns`` with ``ltt%particle_output_every_n_turns``, the
   direct analogue of ``FFILE`` above — which is a separate executable, not
   pytao. SIMBA does not use it today.

   **Ocelot.** ``ocelot.cpbd.track.track()`` takes no turn argument, and
   ``track_nturns``, despite the name, takes a list of single particles and is
   the dynamic-aperture tool, not a ``ParticleArray`` path.

What a Ring Run Reports
-----------------------

Beyond the usual beam and Twiss output:

``ring_parameters()``
    Fractional ``tune_x``/``tune_y``, the periodic ``beta``/``alpha``/``gamma`` in
    both planes, ``stable_x``/``stable_y``, ``closed_orbit_*`` and, where the
    convention allows, ``slip_factor`` and ``momentum_compaction``. Chromaticity is
    deliberately absent from the one-turn map — it needs maps at two momenta, or
    the code's own periodic Twiss, and comes through the optics summary instead.

``one_turn_map``
    The 6x6 map as the code writes it, plus ``one_turn_map_canonical()``, which
    converts it into the ``x, px, y, py, zeta, delta`` convention. The codes do not
    share one: the longitudinal coordinate, its sign and its scaling all differ,
    and each code declares its own in ``otm_convention``.

``run_dynamic_aperture()``
    ``(x, y, turns_survived)`` per point of ``da_rays()``: elegant's
    ``find_aperture`` rays, ``n_lines`` of them from ``+x`` round to ``-x``, each
    with ``nx - 1`` points out to the ellipse through ``(x_max, 0)`` and
    ``(0, y_max)``. ``dynamic_aperture_boundary()`` reduces it to one ``(x, y)``
    per ray: the last survivor before the first loss. A ray that loses nothing
    ends on the ellipse, as in elegant, so make the box larger than the aperture.
    Bmad uses Tao's own angle search instead.

``run_frequency_map()``
    ``(x, y, Qx, Qy, D)`` per surviving grid point, ``D`` being the two-window tune
    diffusion index. Uses NAFF where ``nafflib`` is installed.

    Most codes track the grid for all the turns in one call. Bmad cannot — see the
    note above — so simba writes the grid as an explicit Bmad particle file, tracks
    one turn, and writes the resulting bunch back over the same file for the next.
    The per-particle ``state`` column is written too, so a particle lost on turn 7
    stays lost and the row order still maps each particle to its grid point.

``track_reference_particle()``
    One particle's ``x``/``px``/``y``/``py`` on every turn — what a ring study
    usually wants, and what bunch tracking does not give.

:mod:`simba.Modules.plotting.ring` plots the last three:
``plot_dynamic_aperture``, ``plot_frequency_map`` and ``plot_amplitude_map``, with
``resonance_lines`` to overlay the tune diagram. ``plot_dynamic_aperture`` draws
the boundary (``aperture_boundary``, the same as ``dynamic_aperture_boundary()``),
so every code's plot looks like elegant's. Islands past a loss are left out.

``examples/ring_studies/da_fma.py`` does all of this end to end: it builds a
ten-cell sextupole ring in LAURA, scans it in any of the ring codes and plots the
three maps per code. In short:

.. code-block:: python

    ring = framework["RING"]       # tracking: {turns: 512, dynamic_aperture: {...}}
    ring.preProcess()
    ring.write()
    aperture = ring.run_dynamic_aperture()     # (x, y, turns_survived)
    footprint = ring.run_frequency_map()       # (x, y, Qx, Qy, D)
    boundary = ring.dynamic_aperture_boundary(aperture)

    from simba.Modules.plotting.ring import plot_dynamic_aperture, plot_frequency_map
    plot_dynamic_aperture(aperture, turns=512)
    plot_frequency_map(footprint)

A scan that fails warns and returns an empty list, so don't silence warnings
around one.

.. _which-turn:

Which Turn a Result Came From
-----------------------------

Both the beam and the Twiss objects now carry turn number.

:attr:`~simba.Modules.Beams.beam.turn`
    1-based, and ``None`` where nobody said — a generated distribution, or a file
    written before this existed. A tracked beam always carries a number, including
    on a single-pass line, where it is ``1``. Written into the openPMD file and read
    back, so it survives the round trip:

    .. code-block:: python

        import simba.Modules.Beams as rbf

        beam = rbf.beam()
        rbf.openpmd.read_openpmd_beam_file(beam, "M3.openpmd.hdf5")
        beam.turn     # 1000

:attr:`~simba.Modules.Twiss.twiss.turn`
    A **column**, not a scalar, because the summary merges every line in the run and
    they need not share a turn count. ``0`` is its "nobody said".

The Twiss case needs the column rather than a correction to the numbers, and that is
worth being precise about. A Twiss file holds two kinds of thing. The optics —
``beta_x``, ``mux``, the dispersions — are a property of the lattice and its initial
conditions, and come out the same however many turns were tracked: of the 68 columns
in an Xsuite Twiss file, 57 are bit-identical between a 1-turn and a 4-turn run of
the same input beam. The eleven that move are all bunch statistics — ``sigma_*``,
``mean_*``, ``emit_*n`` — and they come from the particles the *last* turn left
behind. The column is what says which turn that was.

Only :meth:`~simba.Framework.Framework.save_summary_files` fills it. A Twiss file
read on its own has no way to know, because every code writes one per *line*
however many turns ran; the framework knows the turn count, so the stamping happens
there rather than in each of ten per-code readers.
:meth:`~simba.Framework_objects.frameworkLattice.link_handoff_beam` restores the
hand-off between turns from the last turn after the backend has written.

.. _ring-time:

What ``t`` Means
----------------

Every code writes a particle's ``t`` as an **absolute time**, on one clock shared
by all of them. On a ring, a periodic line or a ramp, that clock is the reference
particle's:

.. math::

   T_j(s) = t_0 + \sum_{k<j} \frac{C}{\beta_k c} + \frac{s}{\beta_j c}

for pass :math:`j` (0-based) of a line of length :math:`C`, :math:`\beta_k` being
the reference velocity on pass :math:`k` — constant without a ramp, the ramp's
otherwise. :math:`t_0` is the incoming beam's mean ``t``, fixed when the beam is
read (:meth:`~simba.Framework_objects.frameworkLattice.load_input_beam`), so the
same input gives the same clock whichever code tracks it.

The codes do not keep time the same way: elegant's ``t`` is already absolute, and
the others carry a lag on their own reference instead. Each converts on the way
out.

A code's own coordinate is still there when it is wanted:
:meth:`~simba.Framework_objects.frameworkLattice.native_times` gives it for a
beam the code wrote, and ``time_to_native`` / ``time_from_native`` convert either
way. With :math:`T` the reference clock above,

.. list-table::
   :header-rows: 1
   :widths: 15 15 70

   * - Code
     - Native
     - From ``t``
   * - elegant
     - ``t`` [s]
     - itself
   * - Xsuite
     - ``zeta`` [m]
     - :math:`-\beta_0 c\,(t - T)`
   * - Ocelot
     - ``tau`` [m]
     - :math:`+c\,(t - T)`
   * - MAD-X
     - ``T`` [m]
     - :math:`-c\,(t - T)`

so a particle arriving late has a negative ``zeta`` and ``T`` but a positive
``tau``.

One turn loop
^^^^^^^^^^^^^

Ocelot and MAD-X (where it cannot count turns natively, see
:ref:`native-turns`) track a pass at a time, through one loop
(:meth:`~simba.Framework_objects.frameworkLattice.run_turns`): programs set at the
start of each turn, RF phases moved each pass (see the ``rf`` setting above), only
the last pass of a turn recorded, and the line put back as turn 1 had it at the
end — even if a pass fails — so the optics, a dynamic aperture or a frequency map
run afterwards see the lattice that was asked for, not the last turn's.

The start's own file
^^^^^^^^^^^^^^^^^^^^

A line's start element is where its input beam comes from: ``M1.openpmd.hdf5``
*is* the previous line's end. No code writes over it. Per-turn files at the
start (``M1-t3``) are written as for any other marker.

``input: sample_interval: n`` keeps every *n*-th particle, with the total charge
kept, as the beam is read
(:meth:`~simba.Framework_objects.frameworkLattice.load_input_beam`), so every code
tracks the same particles and none samples on its own.

A sampled beam keeps the full beam's reference: its mean momentum and time,
recorded before sampling
(:attr:`~simba.Framework_objects.frameworkLattice.reference_p0c`,
:attr:`~simba.Framework_objects.frameworkLattice.reference_t0`):

* MAD-X on a linac takes each segment after the first from the beam it is
  given, as it has no other reference past the entrance.
* Bmad takes its reference particle's if the beam has one; it is exact only if
  the sampling keeps it.
* A single Xsuite pass recentres ``zeta`` on the beam's own mean before every
  element, so its ``t`` is still the sample's (by 6e-14 s in the same cell).
* ASTRA, GPT, OPAL, Genesis, Wake_T and CSRTrack still take theirs from the
  beam they are given.

elegant, Ocelot and Cheetah track electrons only
(:attr:`~simba.Framework_objects.frameworkLattice.electrons_only`), and refuse any
other beam as it is read.

Lost particles
^^^^^^^^^^^^^^

A particle a code loses — on an aperture, or to an unstable orbit — is not in the
beam it writes. elegant and MAD-X drop theirs. Xsuite keeps them in its arrays,
marked ``state <= 0`` and frozen where they were lost.

Nothing is kept of *where* or on *which turn* a particle was lost, in any code.
A study that needs a loss map has to get it from the code directly for now:
Xsuite's ``state``, ``at_turn`` and ``s``, or elegant's ``&losses``.

Xsuite also moves the particles it loses to the end of its arrays. The reference
particle (:attr:`~simba.Framework_objects.frameworkLattice.ref_idx`) is now found
by its ``particle_id``.

.. _xsuite-space-charge:

Space charge in Xsuite
----------------------

Xsuite tracks with 3D space charge when the line asks for it the way Ocelot reads
it:

.. code-block:: yaml

    files:
      LINE:
        code: xsuite
        charge: {space_charge_mode: 3d}

Every ``space_charge_step`` (0.1 m, Ocelot's ``unit_step``), an xfields
``SpaceCharge3D`` kick stands for the step around it. Each time the beam passes,
the kick deposits the beam on its grid, solves for the field and kicks it. The
grid is ``getGridSizes(N)`` cells a side, as in Ocelot. Any other mode warns
(:class:`~simba.exceptions.SpaceChargeModeWarning`) and is tracked without.

Things worth knowing:

* **The grids are sized once**, from one pass of (up to 10 000 of) the beam's
  particles without space charge. Kicks with similar beam sizes share a grid.
  A beam that outgrows the last grid by the end of the run warns
  (:class:`~simba.exceptions.SpaceChargeOffGridWarning`). A particle off the grid
  feels no space charge there and adds none to the field.
* **The grids can be re-sized** from a second pass of the sample, through the
  kicks, with ``charge: {space_charge_mode: 3d, space_charge_resize: true}``.
  It costs one more walk of the sample.
* **A kick inside a thick element splits it into slices.** They are the same
  element. Programs and RF phases are bound to it.
* **Single-particle studies see no space charge.** The reference orbit, dynamic
  aperture and frequency map track probes, not the bunch, on
  :attr:`~simba.Codes.Xsuite.Xsuite.xsuiteLattice.single_particle_line`, which
  has the kicks replaced by markers. So does Xsuite's own twiss.
* ``pic_solver`` can be set to ``FFTSolver2p5D`` or ``FFTSolver2p5DAveraged``.
  Both are faster, and both ignore the longitudinal field.

.. note::
   With pyFFTW installed, every xfields solver failed on CPU with an
   ``AssertionError`` at the first kick. xobjects plans its FFTs with pyFFTW, whose
   plan expects the array it was made for, and xfields hands it a new one. SIMBA
   gives each solver a numpy plan instead.

.. _simba-warnings:

simba's Warnings
----------------

Every warning a lattice gives is a class in :mod:`simba.exceptions`, in one of
four groups. A group, or a single warning, can be silenced without matching on
its message text:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Group
     - What it says
   * - :class:`~simba.exceptions.UnsupportedWarning`
     - This code cannot do what was asked, and what it does instead
   * - :class:`~simba.exceptions.SettingWarning`
     - A setting that cannot be read or does not fit the run
   * - :class:`~simba.exceptions.GeometryWarning`
     - The line does not close as a ring
   * - :class:`~simba.exceptions.PhysicsWarning`
     - The run is not the physics that was probably meant

.. code-block:: python

    import warnings
    from simba.exceptions import UnsupportedWarning, NoRadiationWarning

    warnings.simplefilter("ignore", UnsupportedWarning)
    warnings.simplefilter("error", NoRadiationWarning)

All are ``UserWarning`` subclasses, so existing filters keep working. A beam a code
cannot track is :class:`~simba.exceptions.WrongSpeciesError`, which is still a
``ValueError``.

.. _reference-particle:

Tracking the Reference Particle
-------------------------------

All of the above are **methods on the lattice object**, not settings. There is no
``tracking:`` key that turns them on: the block says how the line is tracked, and
these say what to ask of it afterwards. So the file sets ``turns`` and picks a code
that can,

.. code-block:: yaml

    files:
      RING:
        code: xsuite
        tracking:
          turns: 1024

and the trajectory is asked for in Python:

.. code-block:: python

    import simba.Framework as fw

    framework = fw.Framework(machine=machine, directory=outdir, clean=True)
    framework.loadSettings(settings=settings)
    framework.global_parameters["beam"] = beam

    framework.track()

    trajectory = framework["RING"].track_reference_particle()
    # {'x': ..., 'px': ..., 'y': ..., 'py': ...}, each of length `turns`

Three things worth knowing before the first call:

* **The lattice has to exist first.** Xsuite tracks the particle through
  ``self.line`` and elegant writes its own deck beside the lattice file, so call
  this after ``track()`` — or at least after the lattice has been written — rather
  than straight off ``loadSettings()``.
* **A code that cannot do it returns** ``{}`` **rather than raising.** Note the
  asymmetry with ``single_particle`` above: MAD-X is the only ring code that has
  that, and the only one that does not have this.
* **elegant needs a screen.** It reads the trajectory back from a ``WATCH`` file, so
  a line with no ``Screen``, ``Marker`` or ``BPM`` warns and returns nothing.
* **One sample per completed turn**, in every code, so index ``i`` is the end of
  turn ``i + 1`` and the launch condition does not appear. The codes record at
  different places and had to be made to agree: Ocelot's ``p_list`` opens with
  the launch point and is a sample longer than the turn count, and an Xtrack
  monitor samples the *start* of each pass, so neither gave end-of-turn without
  help.

The particle is launched a hair off the closed orbit rather than on it — a hundredth
of the aperture scan's ``x_max``, or :math:`10^{-5}` with no scan configured. One
started exactly *on* the closed orbit repeats itself every turn, which is a perfectly
correct trajectory with no information in it.
