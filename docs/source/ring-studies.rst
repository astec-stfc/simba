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
   that can. See :ref:`ring-capabilities`.

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
screen, holding the last turn, under the unsuffixed name — so a ring run looks like
any other run to everything downstream. With it on you get ``M1-t1`` … ``M1-tN``
*and* the unsuffixed file, which is the one the next section reads by name; see
:ref:`which-turn` for how to tell them apart from the inside.

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
for an element simba has no default for.

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
     - ``name->attribute`` re-stated between turns, which reaches the lattice even
       after ``MAKETHIN``.
   * - Bmad
     - ``set element`` inside the Tao turn loops, which are the only place Bmad
       counts turns here.

.. _ring-capabilities:

What Each Code Can Do
---------------------

A code that cannot honour a setting warns and carries on.

.. list-table::
   :header-rows: 1
   :widths: 15 8 8 10 8 10 10 11 10

   * - Code
     - turns
     - periodic
     - radiation
     - programs
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
     - no
     - no
   * - Xsuite
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
   * - Bmad
     - ring studies only
     - yes
     - yes (on by default)
     - yes
     - yes
     - yes
     - no
     - no

ASTRA, GPT, Cheetah, OPAL and CSRTrack track a line once and have none of these.

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
    ``(x, y, turns_survived)`` per grid point, with
    ``dynamic_aperture_boundary()`` reducing it to the largest surviving ``x`` at
    each ``y``.

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
``resonance_lines`` to overlay the tune diagram.

.. _which-turn:

Which Turn a Result Came From
-----------------------------

A turn number used to live in exactly one place: the **filename**, and only when
``write_turns`` put it there. With the default off, a thousand-turn run writes one
beam file per screen holding the thousandth turn, and nothing in or around that file
said so — on disk it was indistinguishable from a single-turn run.

Both the beam and the Twiss objects now carry it.

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
there rather than in each of the ten per-code readers.

.. note::

   Asking for ``write_turns`` used to **break the handoff between sections**: a code
   routing its end element through per-turn output wrote ``M3-t1`` … ``M3-tN`` and no
   unsuffixed ``M3.openpmd.hdf5``, which is what the next line reads by name.
   :meth:`~simba.Framework_objects.frameworkLattice.link_handoff_beam` restores it
   from the last turn after the backend has written. MAD-X separately used to write
   the end of the line *once* however many turns were tracked — for a ring the one
   place a per-turn record is most wanted — and now writes it per turn as well.

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
