# GROMACS + DFTB+ — QM/MM with separate boundary rules for the QM Hamiltonian and the gradient

> **Branch `develop`.** Everything of `main` plus `GMX_QMMM_ENERGY_CORRECTION`, section 3:
> the reported energy is rebuilt with the rules of the gradient. Switch it off to reproduce
> the energy of `main` bit for bit.

A GROMACS 2021.7 build with the DFTB+ QM/MM interface (DFTB+ coupling by Kubař *et al.*),
PLUMED and a configurable treatment of the QM/MM boundary:

1. **Boundary charge schemes for the QM Hamiltonian** — the charge of the MM atom bonded to
   the QM region can be redistributed (`RC`, `RCD`, `CS`, `AMBER`) in the external potential
   passed to DFTB+, without touching the topology.
2. **Exclusions of the QM–MM gradient** — QM–MM pairs removed up to 1-3 and scaled at 1-4,
   counted along the bond graph or taken from the bonded terms of the topology, with a
   separate rule for the link atoms.
3. **Energy under the rules of the gradient** — the QM–MM electrostatics of the reported
   energy is rebuilt with the rules of the gradient, so that the number and the forces belong
   to the same model.
4. **Boundary treatment in grompp** — which bonded terms at the boundary are removed, and the
   QM–MM Lennard-Jones by the exclusion rules of the force field. Restraints are always kept.
5. **Reports** — `grompp` and `mdrun` write, atom by atom, every term removed from the
   topology and every QM–MM pair removed or scaled in the electrostatics.
6. **Diagnostic output** — the potential on the QM atoms and the QM/MM gradients, split into
   their contributions, for checking against an independent calculation.

Everything is set by environment variables; there is no `.mdp` option and no `tpr` format
change. The potential passed to DFTB+ and the gradient computed by GROMACS follow separate
rules, set independently.

---

## Installation

### GROMACS

An Apptainer/Singularity recipe ships with the repository. It builds DFTB+ 25.1 (with the
API, tblite and s-dftd3), PLUMED 2.9.5 linked in runtime mode with libtorch 1.13.1, and this
GROMACS in double precision with MPI, on Ubuntu 24.04. PLUMED is integrated by its own
`plumed patch -e gromacs-2021.7`.

The recipe clones this repository during the build, so the `.def` file is the only thing you
need locally:

```bash
wget https://raw.githubusercontent.com/Vetrov-Anton/gromacs-dftbplus/develop/Install_gmx_dftb_plumed_torch.def
apptainer build --fakeroot gmx_dftbplus.sif Install_gmx_dftb_plumed_torch.def
```

Versions, the branch to build and the number of build jobs are set in one block at the top
of `%post`; on this branch it is set to `develop`.

The image presets the QM/MM settings below in `%environment`. Each can be overridden per run
with `--env VAR=value`:

| variable | image preset |
|---|---|
| `GMX_QMMM_VARIANT` | `1` (PME) |
| `GMX_QMMM_BONDED_SCHEME` | `classic` |
| `GMX_QMMM_LJ_SCHEME` | `forcefield` |
| `GMX_QMMM_POT_SCHEME` | `CS` |
| `GMX_QMMM_GRAD_EXCL` | `3` |
| `GMX_QMMM_GRAD_LA` | `MM1` |
| `OMP_NUM_THREADS` | `1` |

```bash
apptainer exec --env GMX_QMMM_POT_SCHEME=RCD gmx_dftbplus.sif gmx mdrun ...
```

Because PLUMED is linked at runtime, another PLUMED build can be used without rebuilding
GROMACS, by pointing `PLUMED_KERNEL` at its kernel. The GROMACS QM/MM interface has no
domain decomposition: run on a single rank.

### QMMMtools

The companion module for preparing the QM/MM topology:

```bash
pipx install git+https://github.com/Vetrov-Anton/QMMMtools.git   # command line tool
pip  install git+https://github.com/Vetrov-Anton/QMMMtools.git   # import QMMMtools
```

pipx hides the package from your interpreter, so install it with `pip` as well if you want
`import QMMMtools` to work in a script or a notebook.

---

## Preparing a QM/MM topology: QMMMtools

**[QMMMtools](https://github.com/Vetrov-Anton/QMMMtools)** carves a QM region out of a
finished classical GROMACS system and writes `qm.top`, `qm.gro`, `qm.ndx` and `dftb_in.hsd`.
It places hydrogen link atoms as two-body virtual sites, converts the QM–QM bonds to
connections (`funct 5`) and derives the integer QM charge from the force field.

```bash
qmmmtools prepare -c npt.gro -p topol.top -e '@21313,21391,21408' \
    --solvate 3.0 --hsd --skpath ./3ob-3-1-ophyd/ --mixer anderson
```

A hand-built topology works as well; the four output files and the `qmmmtools check`
consistency test are the intended entry point.

---

## Atoms at the boundary

| name | meaning |
|---|---|
| QM1 | QM atom covalently bonded to an MM atom |
| MM1 | that MM atom |
| MM2 | MM atoms bonded to MM1 (not QM) |
| LA | link atom: a virtual site of the QM group (`virtual_sites2`, funct 1 or 2) constructed from QM1 and MM1 |

The link atoms are recognised from their construction, not from bonds: a QM virtual site
built from one QM and one MM atom is a link atom, and its MM atom is MM1. Connections
(`funct 5`) of the link atoms are not needed and are ignored.

---

## 1. Potential passed to DFTB+: `GMX_QMMM_POT_SCHEME` (mdrun)

The external potential on the QM atoms is computed by GROMACS from the MM point charges and
passed to DFTB+. With a boundary charge scheme, the charge of every MM1 atom is removed from
that potential on all QM atoms and replaced as below; `q0 = q(MM1)/n`, with `n` the number of
MM2 atoms of that MM1. The changes exist **only in the potential passed to DFTB+**: the
topology, the MM–MM interactions and the QM/MM gradient keep the charges of the force field.

| value | potential on the QM atoms |
|---|---|
| `none` (default) | every MM charge in full |
| `RC` | MM1 removed; `q0` at the midpoint of every MM1–MM2 bond (Lin & Truhlar 2005) |
| `RCD` | MM1 removed; `2·q0` at every midpoint, and `q(MM2) − q0` on every MM2 (Lin & Truhlar 2005) |
| `CS` | MM1 removed; `q(MM2) + q0` on every MM2, and a pair `+q0` / `−q0` on the MM1→MM2 line at 0.94 and 1.06 of the bond length (charge shift, Sherwood *et al.* 2003) |
| `AMBER` | MM1 removed; the MM1 charges of a molecule spread evenly over all other MM atoms of the same molecule |

The fictitious charges follow the MM1 and MM2 atoms in every step. With PME they act on the
QM atoms with the full `1/r` (they are not on the grid); with the cut-off variants with the
same kernel as the MM atoms. The `AMBER` shares are added to the charges of the receiving
atoms, on the short-range list and on the PME grid. With PME, the charge of MM1 is removed
in the central cell; its periodic images stay.

mdrun stops with an error if an MM1 atom has no MM2 atom, if an MM2 atom is itself an MM1
atom or is bonded to a QM atom, if one MM1 atom belongs to two link atoms (`RC`, `RCD`, `CS`),
or if the molecule of an MM1 atom has no other MM atom (`AMBER`).

```
QM/MM potential with the boundary charge scheme CS: 1 MM1 charges removed, 6 fictitious point charges added.
```

---

## 2. Exclusions of the QM/MM gradient (mdrun)

The electrostatic QM/MM gradient is computed by GROMACS from the Mulliken charges returned by
DFTB+ and the MM charges of the force field. QM–MM pairs close in the topology are removed
or scaled in it.

### `GMX_QMMM_GRAD_EXCL`

| value | rule |
|---|---|
| `0` | no exclusions |
| `1` | pairs at 1-2 removed |
| `2` | pairs at 1-2 and 1-3 removed |
| `3` (default) | 1-2 and 1-3 removed, 1-4 scaled by `GMX_QMMM_FUDGE_QQ` |
| `BONDED` | pairs taken from the bonded terms of the `tpr`, see below |

With `0`–`3` the bond distance is the shortest path along the chemical bonds of the `tpr`
(bonds, constraints and connections), found by a breadth-first search from every QM atom.

With `BONDED` the pairs come from the bonds, angles and proper dihedrals that remain in the
`tpr` after grompp:

| term | pair | factor |
|---|---|---|
| bond QM–MM | QM, MM | 0 |
| angle QM2–QM1–MM1 | QM2, MM1 | 0 |
| angle QM1–MM1–MM2 | QM1, MM2 | 0 |
| dihedral QM3–QM2–QM1–MM1 | QM3, MM1 | `GMX_QMMM_FUDGE_QQ` |
| dihedral QM2–QM1–MM1–MM2 | QM2, MM2 | `GMX_QMMM_FUDGE_QQ` |
| dihedral QM1–MM1–MM2–MM3 | QM1, MM3 | `GMX_QMMM_FUDGE_QQ` |

Improper dihedrals are not used. A pair found both removed and scaled is removed. The terms
removed by `GMX_QMMM_BONDED_SCHEME=classic` (section 4) are absent from the `tpr`, so their
pairs keep the full interaction. With QM/MM, grompp keeps the angles and proper dihedrals whose
parameters are all zero, so that every bonded quadruple is present; they contribute nothing
to the energy.

### `GMX_QMMM_GRAD_LA`

| value | link atoms in the gradient |
|---|---|
| `MM1` (default) | as if the link atom were its MM1 atom: the pair with MM1 is removed, and the pairs with the other MM atoms follow the rule above counted from MM1 |
| `QM1` | the same pairs and factors as its QM1 atom |
| `exclude` | the link atoms take no part in the gradient at all: their charge is zero there, and its sum is spread evenly over the MM atoms of their own molecule, so that the charge of the system is unchanged |

With `exclude` the link atoms are also absent from the gradient of the QM periodic images, and no
pairs are excluded for them (there is nothing to exclude with zero charge), so the pair counters
drop accordingly. The spread is recomputed in every step from the current Mulliken charges:

```
QM/MM gradient with GMX_QMMM_GRAD_LA=exclude: molecule of atoms 1-3743, the charge of its 10 link atoms
  is removed from the gradient and spread over its 3634 MM atoms in every step.
```

The potential of section 1 is not affected by this, so the charges and the number of SCC
iterations on a given geometry are the same as with `MM1`. The forces are then no longer the
derivative of the energy of the Hamiltonian, which is the price of this option; section 3 brings
the reported energy to the same rules, but the forces stay what they are.

### `GMX_QMMM_FUDGE_QQ`

The factor of the 1-4 pairs. Defaults to `fudgeQQ` of the force field (0.8333 for AMBER).

### PME

With `GMX_QMMM_VARIANT=1` the reciprocal-space part is evaluated on a grid over all MM
charges and cannot be scaled there. For every removed or scaled pair the fraction `(1−s)` of
its reciprocal-space term `erf(βr)/r` is subtracted as a pair term; together with the scaled
real-space term `s·erfc(βr)/r` this gives `s/r` for the pair. The same construction is used
for the MM1 charges removed from the potential in section 1.

```
QM/MM gradient: GMX_QMMM_GRAD_EXCL = 3, 11 QM--MM pairs removed, 16 scaled with GMX_QMMM_FUDGE_QQ = 0.8333;
  1 link atoms, GMX_QMMM_GRAD_LA = MM1.
```

The energy that belongs to this gradient is the subject of section 3.

---

## 3. Energy under the rules of the gradient: `GMX_QMMM_ENERGY_CORRECTION` (mdrun)

DFTB+ returns an energy whose QM–MM electrostatics is the one of the QM Hamiltonian, i.e. of the
potential of section 1, while the forces are built with the rules of section 2. As soon as the two
sets of rules differ, the reported energy and the reported forces belong to different models. The
correction replaces the first contribution by the second:

```
E = E(DFTB+) − Σ_A q_A φ_pot(A) + Σ_A q_A φ_grad(A)
```

The charges, the exclusions, the 1-4 factor and the link-atom treatment of the added term are
exactly those of the gradient, the redistribution of `GMX_QMMM_GRAD_LA=exclude` included. With
`GMX_QMMM_VARIANT=1` the contribution of the periodic images of the QM charges is rebuilt as well;
this costs two extra PME calls per step and is done only when the two charge sets differ.

| value | QM–MM electrostatics of the reported energy |
|---|---|
| `on` (default) | the rules of the gradient, section 2 |
| `off` | the rules of the QM Hamiltonian, section 1 |

The correction is identically zero when the two sets of rules coincide, and with `off` the energy
is the one the code produced before this option existed. The step costs 6–8 % more with the
correction switched on.

```
QM/MM energy: the QM--MM electrostatics of the reported energy follows the rules of the gradient
  (GMX_QMMM_ENERGY_CORRECTION = on).
```

**What it does not do.** The correction changes the energy and never the forces. With different
rules for the potential and for the gradient the forces are not the derivative of any function:
what is missing from them is the charge response `Σ_A (φ_grad − φ_pot)(A) · dq_A/dR`, and no
function of the coordinates added to the energy can produce it. The correction therefore makes the
energy consistent with the force model and comparable between schemes, but it does not restore the
conservation of energy in NVE — measured on a solvated tripeptide over 200 ps at 0.5 fs, the drift
goes from 440 to 408 kJ/(mol·ns) for `CS` with `GRAD_EXCL=3` and `GRAD_LA=MM1`, both figures far
above the 57 of matched rules. Where conservation matters, match the rules: `POT_SCHEME=none`
with `GRAD_EXCL=0`.

---

## 4. Topology at the boundary (grompp)

### `GMX_QMMM_BONDED_SCHEME`

| value | bonded terms removed |
|---|---|
| `classic` (default) | every term in which all but one atom are QM (so a QM–QM–MM angle and a QM–QM–QM–MM dihedral go), because the QM calculation with the link atom describes it |
| `amber` | only terms whose atoms are all QM; every term with an MM atom is kept at the force-field level |

Bonds between two QM atoms are converted to connections (`F_CONNBONDS`), so the connectivity
stays complete. The scheme is written into the `tpr`; mdrun only reminds you of that if it
sees the variable.

### `GMX_QMMM_LJ_SCHEME`

| value | Lennard-Jones between QM and MM atoms |
|---|---|
| `forcefield` (default) | by the exclusion rules of the force field (`nrexcl`, `[ pairs ]`); only the LJ within the QM region is excluded |
| `exclude` | in addition, the LJ and LJ-14 of every QM atom with the MM atoms bonded to the QM region are excluded (`classic` only) |

No LJ exclusions are generated for the link atoms; give them zero LJ parameters.

### Restraints

Position, flat-bottomed position, distance, orientation, angle and dihedral restraints and
restraint potentials are always kept, also on QM atoms.

### Summary printed by grompp

```
QM/MM: force-field terms removed with the 'classic' scheme, by interaction type:
  interaction                all-QM     QM--MM    MM-only
  Angle                          21          3          0
  Proper Dih.                    26          6          0
  Improper Dih.                   2          0          0
  LJ-14                          20          0          0
  total                          69          9          0
```

`all-QM` terms are described by the QM calculation, `QM--MM` terms span the boundary. The
`LJ-14` row counts `[ pairs ]` entries (1-4 LJ and 1-4 Coulomb together); pairs of two QM
atoms are always removed.

---

## 5. Reports

| file | written by | content |
|---|---|---|
| `qmmm_topology_report.txt` | grompp | QM atoms, boundary bonds, every removed bonded term, the QM–QM bonds converted to connections, removed LJ-14 pairs, LJ exclusions, and the final QM–MM LJ exclusions and LJ-14 pairs of the `tpr` |
| `qmmm_exclusion_report.txt` | mdrun | QM atoms, link atoms, the MM1 atoms removed from the potential, the `AMBER` shares, the fictitious charges, every QM–MM pair removed or scaled in the gradient with the rule that produced it, and the QM–MM LJ exclusions and LJ-14 pairs of the `tpr` |

Atom numbers are global and 1-based, as in the `.gro` file, with residue and atom names. The
file names can be changed with `GMX_QMMM_TOPOLOGY_REPORT` and `GMX_QMMM_EXCLUSION_REPORT`;
`GMX_QMMM_REPORTS=off` (or `0`, `no`, `false`) switches both reports off.

---

## 6. QM/MM electrostatics variant

| variable | meaning |
|---|---|
| `GMX_QMMM_VARIANT=0` | no electrostatic QM/MM interaction (vacuum QM) — the default when unset |
| `GMX_QMMM_VARIANT=1` | PME (requires a periodic system) |
| `GMX_QMMM_VARIANT=2` | switched cut-off |
| `GMX_QMMM_VARIANT=3` | reaction field |
| `GMX_QMMM_VARIANT=4` | shifted cut-off |
| `GMX_QMMM_PME_DIPCOR` | dipole (surface) correction for PME — disabled: with `GMX_QMMM_VARIANT=1` mdrun prints why and exits; with any other variant it is ignored |

All boundary schemes of sections 1–3 work with every variant.

---

## 7. DFTB output files (mdrun)

Each of the following is set to an **integer stride in steps**; the file is opened in append
mode in the run directory on step 0 and written every *N* steps. Unset means no file. The
variables that describe the MM environment are ignored when `GMX_QMMM_VARIANT=0`.

| variable | file | content |
|---|---|---|
| `GMX_DFTB_CHARGES=N` | `qm_dftb_charges.xvg` | Mulliken charges of the QM atoms |
| `GMX_DFTB_ESP=N` | `qm_dftb_esp.xvg` | potential on each QM atom, MM and QM images summed |
| `GMX_DFTB_ESP_SPLIT=N` | `qm_dftb_esp_split.xvg` | the same potential with the two contributions kept apart |
| `GMX_DFTB_QM_COORD=N` | `qm_dftb_qm.qxyz` | QM coordinates and charges (XYZQ) |
| `GMX_DFTB_MM_COORD=N` | `qm_dftb_mm.qxyz` | coordinates and charges of the MM atoms **on the short-range list** |
| `GMX_DFTB_MM_COORD_FULL=N` | `qm_dftb_mm_full.qxyz` | coordinates and charges of **all** MM atoms |
| `GMX_DFTB_QMMM_GRAD=N` | `qm_dftb_grad.xvg` | gradients on the QM atoms and on the short-range MM atoms |
| `GMX_DFTB_QMMM_GRAD_FULL=N` | `qm_dftb_grad_full.xvg` | gradients on **all** MM atoms (PME only) |
| `GMX_DFTB_ENERGY_CORR=N` | `qm_dftb_energy_corr.xvg` | the correction of section 3 and the corrected QM energy, kJ/mol |

`GMX_DFTB_MM_COORD_FULL` and `GMX_DFTB_QMMM_GRAD_FULL` on a solvated system write the whole
box every *N* steps — pick a large stride.

### Splitting the potential on the QM atoms

The external potential has two sources: the MM point charges (with the scheme of section 1)
and — with PME — the periodic images of the QM charges. `GMX_DFTB_ESP_SPLIT` writes one row
per written step, the step number followed by three numbers per QM atom, in volts:

```
       0  2.22433  1.07294  3.29727  3.58481  1.07115  4.65596  2.56784 ...
           |        |        |
           |        |        the sum, i.e. what qm_dftb_esp.xvg contains
           |        the periodic images of the QM charges
           the MM atoms
```

The second number is zero unless `GMX_QMMM_VARIANT=1`. With PME the first number is the
Ewald potential of the MM subsystem, periodic images of the MM charges included; the cut-off
variants give a plain pair sum.

### Gradients

`GMX_DFTB_QMMM_GRAD` writes the gradients before the QM gradient from DFTB+ and the
electrostatic gradient computed by GROMACS are added together — one block for the QM atoms
and one for the MM atoms of the short-range list:

```
QM gradients: DFTB+, electrostatic, total (hartree/bohr) step 0
QM     1   -0.0420499  -0.0223395   0.0009268    0.0007340  -0.0012573  -0.0017847   -0.0413159  -0.0235968  -0.0008579
...
MM gradients on the short-range list (hartree/bohr) step 0
MM     3        3    0.0000568   0.0026999  -0.0005152
```

A QM row carries the running number of the QM atom (the order of `qm_dftb_qm.qxyz` and
`qm_dftb_charges.xvg`) and three vectors: the gradient returned by DFTB+, the electrostatic
gradient due to the environment (with the exclusions of section 2), and their sum. An MM row
carries the running number on the short-range list and the global atom number (1-based, as
in the `.gro`). Units are hartree/bohr; these are gradients, the force is their negative.
Multiply by `HARTREE_BOHR2MD` ≈ 4.96147·10⁴ for kJ/mol/nm. With PME the electrostatic column
also contains the periodic images of the QM charges.

`GMX_DFTB_QMMM_GRAD_FULL` adds the gradients on all MM atoms (PME only), one row per atom with
the global atom number. An MM atom of the short-range list appears in both files, and its
gradient is the sum of the two entries.

---

## Environment variables

| variable | read by | values | code default | image preset |
|---|---|---|---|---|
| `GMX_QMMM_VARIANT` | mdrun | `0`–`4` | `0` | `1` |
| `GMX_QMMM_BONDED_SCHEME` | grompp | `classic`, `amber` | `classic` | `classic` |
| `GMX_QMMM_LJ_SCHEME` | grompp | `forcefield`, `exclude` | `forcefield` | `forcefield` |
| `GMX_QMMM_POT_SCHEME` | mdrun | `none`, `RC`, `RCD`, `CS`, `AMBER` | `none` | `CS` |
| `GMX_QMMM_GRAD_EXCL` | mdrun | `0`–`3`, `BONDED` | `3` | `3` |
| `GMX_QMMM_GRAD_LA` | mdrun | `MM1`, `QM1`, `exclude` | `MM1` | `MM1` |
| `GMX_QMMM_FUDGE_QQ` | mdrun | float | force-field `fudgeQQ` | — |
| `GMX_QMMM_ENERGY_CORRECTION` | mdrun | `on`, `off` | `on` | — |
| `GMX_QMMM_REPORTS` | grompp, mdrun | `off`, `0`, `no`, `false` | on | — |
| `GMX_QMMM_TOPOLOGY_REPORT` | grompp | file name | `qmmm_topology_report.txt` | — |
| `GMX_QMMM_EXCLUSION_REPORT` | mdrun | file name | `qmmm_exclusion_report.txt` | — |
| `GMX_QMMM_PME_DIPCOR` | mdrun | set/unset | unset (disabled) | — |
| `GMX_DFTB_CHARGES` | mdrun | stride in steps | off | — |
| `GMX_DFTB_ESP` | mdrun | stride in steps | off | — |
| `GMX_DFTB_ESP_SPLIT` | mdrun | stride in steps | off | — |
| `GMX_DFTB_QM_COORD` | mdrun | stride in steps | off | — |
| `GMX_DFTB_MM_COORD` | mdrun | stride in steps | off | — |
| `GMX_DFTB_MM_COORD_FULL` | mdrun | stride in steps | off | — |
| `GMX_DFTB_QMMM_GRAD` | mdrun | stride in steps | off | — |
| `GMX_DFTB_QMMM_GRAD_FULL` | mdrun | stride in steps | off | — |
| `GMX_DFTB_ENERGY_CORR` | mdrun | stride in steps | off | — |

An unknown value of `GMX_QMMM_BONDED_SCHEME`, `GMX_QMMM_LJ_SCHEME`, `GMX_QMMM_POT_SCHEME`,
`GMX_QMMM_GRAD_EXCL`, `GMX_QMMM_GRAD_LA`, `GMX_QMMM_FUDGE_QQ` or `GMX_QMMM_ENERGY_CORRECTION`
is a fatal error.

---

## References

- H. Lin, D. G. Truhlar, *J. Phys. Chem. A* **109**, 3991 (2005) — RC and RCD schemes.
- P. Sherwood *et al.*, *J. Mol. Struct. THEOCHEM* **632**, 1 (2003) — ChemShell, charge shift.
- T. Kubař, K. Welke, G. Groenhof, *J. Comput. Chem.* **36**, 1978 (2015) — DFTB3 QM/MM in GROMACS.

## Licence

GROMACS is distributed under the GNU LGPL v2.1; see `COPYING`. The modifications described
here are released under the same terms.
