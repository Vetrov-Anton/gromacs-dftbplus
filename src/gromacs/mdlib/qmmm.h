/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 1991-2000, University of Groningen, The Netherlands.
 * Copyright (c) 2001-2008, The GROMACS development team.
 * Copyright (c) 2013,2014,2015,2017,2018 by the GROMACS development team.
 * Copyright (c) 2019,2020, by the GROMACS development team, led by
 * Mark Abraham, David van der Spoel, Berk Hess, and Erik Lindahl,
 * and including many others, as listed in the AUTHORS file in the
 * top-level source directory and at http://www.gromacs.org.
 *
 * GROMACS is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 2.1
 * of the License, or (at your option) any later version.
 *
 * GROMACS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with GROMACS; if not, see
 * http://www.gnu.org/licenses, or write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
 *
 * If you want to redistribute modifications to GROMACS, please
 * consider that scientific software is very special. Version
 * control is crucial - bugs must be traceable. We will be happy to
 * consider code for inclusion in the official distribution, but
 * derived work must not be called official GROMACS. Details are found
 * in the README & COPYING files - if they are missing, get the
 * official version at http://www.gromacs.org.
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out http://www.gromacs.org.
 */
#ifndef GMX_MDLIB_QMMM_H
#define GMX_MDLIB_QMMM_H

#include "config.h"

#include <utility>
#include <vector>

#include "gromacs/math/paddedvector.h"
#include "gromacs/math/vectypes.h"
#include "gromacs/mdlib/tgroup.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/timing/wallcycle.h"

//#include "gromacs/mdlib/qm_dftbplus.h"
//#include "gromacs/mdlib/qm_gamess.h"
//#include "gromacs/mdlib/qm_gaussian.h"
//#include "gromacs/mdlib/qm_mopac.h"
//#include "gromacs/mdlib/qm_orca.h"

#define GMX_QMMM (GMX_QMMM_MOPAC || GMX_QMMM_GAMESS || GMX_QMMM_GAUSSIAN || GMX_QMMM_ORCA || GMX_QMMM_DFTBPLUS)

struct t_nrnb;
struct nonbonded_verlet_t;

struct gmx_pme_t;
enum class PbcType : int;

struct DftbPlus;
struct Context;

// THIS STRUCTURE IS TENTATIVE,
// JUST FOR THE BEGINNING.
// MANY THINGS ARE STORED TWICE!
// TO BE CLEANED UP!
// Also, it would be cool to rename the structure.
class QMMM_PME {
private:
public:
  PaddedVector<gmx::RVec> x;
  PaddedVector<gmx::RVec> f;
  std::vector<real>       q;
  matrix                  vir;
  real*                   pot; // electrostatic potential from PME ("external shift" in DFTB),
  gmx_bool                surf_corr_pme; // whether the surface correction shall be considered or not (if not = tin-foil boundary cond.)
  real                    epsilon_r; // if yes, this is the dielectric constant to be considered

  rvec*                   qmgrad; // gradients at QM atoms
  rvec*                   mmgrad; // gradients at external charges
  rvec*                   partmmgrad; // temp. array for components of gradients at external charges
  rvec                    com; // center of mass
  real*                   mass; // masses of the atoms
  real                    inv_tot_mass; // 1 / sum(mass)
  real*                   ze; // magnitudes of external charges, NNDIM
  int                     n; // number of QM atoms
  int                     ne; // number of external charges
  int                     cutoff_qmmm;     // whether a switched cut-off QM/MM calculation shall be done instead of PME
  // output
  int                     output_qm_freq;  // how often (if ever) the QM coordinates shall be written
  int                     output_mm_freq;  // how often (if ever) the MM coordinates shall be written
  int                     output_nbl_freq; // how often (if ever) the short-range MM coordinates shall be written
  // for PME
  double                  rcoulomb_pme; // cut-off
  double                  rlist_pme;    // neighborlist cut-off (for PME, equal to rcoulomb_pme; for switched cut-off, larger)
  int                     nstlist_pme;  // frequency of neighborsearching
  int                     lastlist_pme; // last step when neighborsearching was done
} ;

struct gmx_localtop_t;
struct gmx_mtop_t;
struct t_commrec;
struct t_forcerec;
struct t_inputrec;
struct t_mdatoms;

namespace gmx
{
//class ForceWithShiftForces;
class ForceWithVirial;
}

class QMMM_QMrec;
class QMMM_MMrec;

class QMMM_QMgaussian {
private:
public:
    int      nQMcpus;  // no. of CPUs used for the QM calc.
    int      QMmem;    // memory for the gaussian calc.
    int      accuracy; // convergence criterium (E(-x))
    gmx_bool cpmcscf;  // using cpmcscf(l1003)
    char*    gauss_dir;
    char*    gauss_exe;
    char*    devel_dir;

/*! \brief
 * Initialize gaussian datastructures.
 */
void init_gaussian();

/*! \brief
 * Call gaussian to do qm calculation.
 *
 * \param[in] fr Global forcerec.
 * \param[in] qm QM part of forcerec.
 * \param[in] mm mm part of forcerec.
 * \param[in] f  force vector.
 * \param[in] fshift shift of force vector.
 */
real call_gaussian(const QMMM_QMrec& qm,
                   const QMMM_MMrec& mm,
                   rvec              f[],
                   rvec              fshift[]) const;
} ;

class QMMM_QMrec {
private:
    int     nrQMatoms;      // total nr of QM atoms
    rvec*   xQM;            // shifted to center of box
    int*    indexQM;        // atom i = atom indexQM[i] in mdrun
    int*    atomicnumberQM; // atomic numbers of QM atoms
    real*   QMcharges;      // atomic charges of QM atoms(ONIOM)
    int*    shiftQM;
    int     QMcharge;       // charge of the QM system
    int     multiplicity;   // multipicity (no of unpaired eln)
    int     QMmethod;       // see enums.h for all methods
    int     QMbasis;        // see enums.h for all bases
    int     nelectrons;     // total number of elecs in QM region
    int     CASelectrons;   // electron in active space
    int     CASorbitals;    // orbitals in active space

    matrix  box;
    int     qmmm_variant;
    real    rcoulomb;
    real    ewaldcoeff_q;
    real    epsilon_r;
    double* pot_qmmm;       // electric potential induced by the MM atoms
    double* pot_qmqm;       // el. pot. induced by the periodic images of the QM atoms

    void init_QMrec(int               grpnr,
                    int               nr,
                    const int*        atomarray,
                    const gmx_mtop_t* mtop,
                    const t_inputrec* ir);

    friend class QMMM_rec;
    friend void init_dftbplus(QMMM_QMrec*       qm,
                              const t_forcerec* fr,
                              const t_inputrec* ir,
                              const t_commrec*  cr,
                              gmx_wallcycle_t   wcycle);
    friend real call_dftbplus(const t_forcerec* fr,
                              const t_commrec*  cr,
                              QMMM_QMrec*       qm,
                              QMMM_MMrec*       mm,
                              rvec              f[],
                              rvec              fshift[],
                              t_nrnb*           nrnb,
                              gmx_wallcycle_t   wcycle);

public:
    DftbPlus        *dpcalc;        // DFTB+ calculator
    Context         *dftbContext;   // some data for DFTB+, referenced to by DFTB through *dpcalc

    QMMM_QMgaussian  gaussian;

    char            *orca_basename; // basename for I/O with orca
    char            *orca_dir;      // directory for ORCA

    // output
    int              nrQMatoms_get()const;
    int              qmmm_variant_get()const;
    double           xQM_get(int atom, int coordinate)const;
    real             QMcharges_get(int atom)const;
    double           pot_qmmm_get(int atom)const;
    double           pot_qmqm_get(int atom)const;
    int              atomicnumberQM_get(int atom)const;
    int              QMcharge_get()const;
    int              multiplicity_get()const;
    int              QMmethod_get()const;
    int              QMbasis_get()const;
    int              nelectrons_get()const;
    int              CASelectrons_get()const;
    int              CASorbitals_get()const;
    real             box_xx_get()const;
    real             box_yy_get()const;
    real             box_zz_get()const;
    // input
    void             QMcharges_set(int atom, real value);
    void             pot_qmmm_set(int atom, double value);
    void             pot_qmqm_set(int atom, double value);
} ;

class QMMM_MMrec {
private:
public:
    real           scalefactor;

    // There are 3 kinds of MM atom lists.
    // (1) the short-range list that is updated in every step of MD,
    //    and is used in the QM calculation:
    int                     nrMMatoms; // nr of MM atoms
    PaddedVector<gmx::RVec> xMM;       // coordinates shifted to the center of the box
    std::vector<int>        indexMM;   // atom i = atom indexMM[i] in mdrun
    std::vector<real>       MMcharges; // magnitude of MM point charges
    std::vector<int>        shiftMM;

    // (2) the short-range list that is produced
    //    by the (group or Verlet) neighborsearching procedure,
    //    and is updated in every neighborsearching step.
    //    The processing of this list in every step of MD
    //    yields the list under (1).
    //    This list itself is not used in QM calculation directly:
    int              nrMMatoms_nbl;
    std::vector<int> indexMM_nbl;
    std::vector<int> shiftMM_nbl;

    // (3) the list of *all* of the non-QM atoms,
    //    which is static throughout the simulation and never needs to be updated:
    int                     nrMMatoms_full;
    PaddedVector<gmx::RVec> xMM_full;
    std::vector<int>        indexMM_full;
    std::vector<real>       MMcharges_full;
    std::vector<int>        shiftMM_full;

    // Scaling of the QM--MM electrostatic interaction, see QMMM_rec::init_QMMM_exclusions().
    // The external potential passed to DFTB+ and the QM/MM gradient use separate factors:
    //   qmmmScalePot  -- the MM1 atoms zeroed by a boundary charge scheme (GMX_QMMM_POT_SCHEME);
    //   qmmmScaleGrad -- the exclusions and scalings of the gradient (GMX_QMMM_GRAD_EXCL).
    // Both are indexed as [j * nrMMatoms + k] for QM atom j and MM atom k of the
    //   short-range list (1), and left empty when there is nothing to scale;
    // localIndexOfAtom maps a global atom number onto its position
    //   in the short-range list (1), or -1 if it is not on that list.
    std::vector<real> qmmmScalePot;
    std::vector<real> qmmmScaleGrad;
    std::vector<int>  localIndexOfAtom;

    //! Factor of the charge of MM atom \p k in the external potential on QM atom \p j
    real qmmmScaleFactorPot(int j, int k) const
    {
        return qmmmScalePot.empty() ? real(1.0)
                                    : qmmmScalePot[static_cast<size_t>(j) * nrMMatoms + k];
    }
    //! Factor of the QM/MM gradient between QM atom \p j and MM atom \p k
    real qmmmScaleFactorGrad(int j, int k) const
    {
        return qmmmScaleGrad.empty() ? real(1.0)
                                     : qmmmScaleGrad[static_cast<size_t>(j) * nrMMatoms + k];
    }

    void init_MMrec(real       scalefactor_in,
                    int        nrMMatoms_full_in,
                    int        natoms,
                    int        nrQMatoms,
                    const int* indexQM,
                    int*       found_mm_atoms);
} ;

class QMMM_rec {
private:
public:

 // int                      nrQMlayers; // number of QM groups/layers (total layers +1 (MM))
    std::vector<QMMM_QMrec>  qm;         // atoms and run params for each QM group
    std::vector<QMMM_MMrec>  mm;         // there can only be one MM subsystem !
    std::vector<QMMM_PME>    pme;        // [0] == pme_full, [1] == pme_qmonly
    PbcType                  pbcType;
    struct gmx_pme_t* const* pmedata;

    // Treatment of the QM/MM boundary in the QM--MM electrostatics, see init_QMMM_exclusions().
    //
    // External potential passed to DFTB+ (GMX_QMMM_POT_SCHEME): with a boundary charge
    //   scheme the charge of every MM1 atom is removed from the potential on all QM atoms,
    //   and fictitious point charges near MM1 and MM2 are added. Both are seen by the QM
    //   atoms only; the topology, the MM interactions and the gradient are not affected.
    enum class PotScheme
    {
        None, // every MM charge enters the potential in full
        RC,   // redistributed charge: q(MM1)/n on the midpoints of the MM1-MM2 bonds
        RCD,  // redistributed charge and dipole: 2q(MM1)/n there, q(MM2) - q(MM1)/n
        CS,   // charge shift: q(MM2) + q(MM1)/n, and +-q(MM1)/n at 0.94 and 1.06 of MM1->MM2
        Amber // as in AMBER: the MM1 charges spread evenly over the other MM atoms of their molecule
    };
    PotScheme potScheme = PotScheme::None;
    // With PotScheme::Amber, the charge added to each atom (global index) in the potential
    //   only, before mm[0].scalefactor: the MM1 charges of a molecule divided by the number
    //   of its other MM atoms, on those atoms; zero elsewhere. Empty with the other schemes.
    std::vector<real> potChargeShift;
    //! Charge added to global atom \p a in the potential (without mm[0].scalefactor)
    real potChargeShiftOf(int a) const { return potChargeShift.empty() ? real(0.0) : potChargeShift[a]; }
    // A fictitious point charge of the potential: charge q at x(a) + f * (x(b) - x(a)),
    //   a and b global atom numbers (a = MM1, b = MM2)
    struct PotPoint
    {
        int  a;
        int  b;
        real f;
        real q;
    };
    std::vector<PotPoint> potPoints;
    // For every QM atom, the (global MM atom index, factor) pairs of the potential
    //   and of the gradient. Built once at initialization.
    std::vector<std::vector<std::pair<int, real>>> mmScalePot;
    std::vector<std::vector<std::pair<int, real>>> mmScaleGrad;

    // Exclusions of the QM/MM gradient (GMX_QMMM_GRAD_EXCL): gradExcl = 0..3 bonds along
    //   the bond graph, or gradBonded to take them from the bonds, angles and proper
    //   dihedrals of the topology; 1-4 pairs are scaled with gradFudgeQQ (GMX_QMMM_FUDGE_QQ).
    int  gradExcl    = 3;
    bool gradBonded  = false;
    real gradFudgeQQ = 1.0;
    // Link atoms in the gradient (GMX_QMMM_GRAD_LA): they are excluded as if they were
    //   their QM1 atom or, by default, their MM1 atom; or they are taken out of the QM/MM
    //   electrostatic gradient altogether (exclude), their charge being spread evenly over
    //   the MM atoms of their own molecule so that the total charge is preserved.
    enum class GradLa
    {
        QM1,    // the exclusions of the QM1 atom
        MM1,    // the exclusions counted from the MM1 atom (default)
        Exclude // zero charge in the gradient, spread over the MM atoms of the molecule
    };
    GradLa gradLa = GradLa::MM1;
    //! Whether the link atoms are excluded as their MM1 atom
    bool gradLaAsMM1() const { return gradLa == GradLa::MM1; }
    //! Whether the link atoms carry no charge in the QM/MM gradient
    bool gradLaExcluded() const { return gradLa == GradLa::Exclude; }

    // GradLa::Exclude. The molecules that contain link atoms: for each of them the link
    //   atoms (as indices of the QM list) and the MM atoms of the same molecule that
    //   receive their charge (global atom indices). Built once at initialization.
    struct GradLaMolecule
    {
        std::vector<int> linkAtomsOfQmList;
        std::vector<int> receivers;
    };
    std::vector<GradLaMolecule> gradLaMolecules;
    // Global atom -> its molecule in gradLaMolecules, or -1. Empty unless GradLa::Exclude.
    std::vector<int> gradLaMolOfAtom;
    // Charges of the QM/MM gradient, rebuilt in every step by update_gradient_charges()
    //   when the link atoms are excluded: the Mulliken charges with the link atoms zeroed,
    //   and the MM charges of the short-range and of the full list with the charge of the
    //   link atoms added evenly (incl. mm[0].scalefactor). Empty otherwise, and the plain
    //   charges are used then.
    std::vector<real> gradChargesQM;
    std::vector<real> gradChargesMM;
    std::vector<real> gradChargesMMfull;
    //! Charge of QM atom \p j in the QM/MM gradient and in the QM periodic images
    real gradChargeQM(int j) const { return gradChargesQM.empty() ? qm[0].QMcharges[j] : gradChargesQM[j]; }
    //! Charge of MM atom \p k of the short-range list in the QM/MM gradient
    real gradChargeMM(int k) const { return gradChargesMM.empty() ? mm[0].MMcharges[k] : gradChargesMM[k]; }
    //! Charge of MM atom \p k of the full list in the QM/MM gradient
    real gradChargeMMfull(int k) const
    {
        return gradChargesMMfull.empty() ? mm[0].MMcharges_full[k] : gradChargesMMfull[k];
    }
    // Rebuild the charges above from the current Mulliken charges and MM list.
    //   Called at the beginning of gradient_QM_MM(); does nothing unless GradLa::Exclude.
    void update_gradient_charges(int variant);

    // Add the fictitious point charges of the boundary scheme to the potential
    //   on the QM atoms (in e/nm, before the conversion to atomic units).
    void add_boundary_scheme_potential(int variant, real* pot);

    // AMBER: the MM charges of the potential on the current short-range list (topology charge
    //   plus shift, incl. scalefactor), rebuilt with the list; and the shifts on the full MM
    //   list (index of xMM_full, incl. scalefactor), built on first use. Empty otherwise.
    std::vector<real> potChargesSR;
    std::vector<real> potShiftFull;
    // Global atoms of the previous short-range list, to reset localIndexOfAtom cheaply.
    std::vector<int> previousIndexMM;

    QMMM_rec(const t_commrec*                 cr,
             const gmx_mtop_t*                mtop,
             const t_inputrec*                ir,
             const t_forcerec*                fr);
          // const gmx_wallcycle_t gmx_unused wcycle);
    // From topology->atoms.atomname and topology->atoms.atomtype
    //   the atom names and types are read;
    // From inputrec->QMcharge resp. inputrec->QMmult the nelecs and multiplicity are determined
    //   and md->cQMMM gives numbers of the MM and QM atoms

    /*
    void update_QMMMrec(const t_commrec*  cr,
                        const t_forcerec* fr,
                        const rvec*       x,
                        const t_mdatoms*  md,
                        const matrix      box);
    */

    // update_QMMMrec() fills the MM stuff in QMMMrec.
    // The MM atoms are taken from the neighbourlists of the QM atoms.
    // In a QMMM run, this routine should be called at every step,
    //   since it updates the MM elements of the t_QMMMrec struct.
    // CORRECTION: only call it in every NS step,
    // to process the newly created neighborlists!

    void update_QMMMrec_verlet_ns(const t_commrec*    cr,
                                  nonbonded_verlet_t* nbv,
                                  const rvec          x[],
                                  const t_mdatoms*    md,
                                  const matrix        box);
    
    void update_QMMMrec_dftb(const t_commrec*  cr,
                             rvec*             shift_vec,
                             const rvec        x[],
                             const t_mdatoms*  md,
                             const matrix      box);
    
    // Set up the topological exclusions of the QM--MM electrostatics.
    // Called once, from the constructor.
    void init_QMMM_exclusions(const gmx_mtop_t* mtop, const t_forcerec* fr, const t_commrec* cr);

    // Fill mm[0].qmmmScalePot and mm[0].qmmmScaleGrad for the current short-range MM list.
    // Called from update_QMMM_coord(), i.e. in every step.
    void update_QMMM_exclusion_scaling(int natoms);

    void update_QMMM_coord(const t_commrec*  cr,
                           rvec*             shift_vec,
                           const rvec        x[],
                           const t_mdatoms*  md,
                           const matrix      box);
    
    // New routines for QM/MM interactions
    
    void calculate_SR_QM_MM(int   variant,
                            real* pot);
    
    void calculate_LR_QM_MM(const t_commrec*  cr,
                            t_nrnb*           nrnb,
                            gmx_wallcycle_t   wcycle,
                            struct gmx_pme_t* pmedata,
                            real*             pot);
    
    void calculate_complete_QM_QM(const t_commrec*  cr,
                                  t_nrnb*           nrnb,
                                  gmx_wallcycle_t   wcycle,
                                  struct gmx_pme_t* pmedata,
                                  real*             pot);
    
    void gradient_QM_MM(const t_commrec*  cr,
                        t_nrnb*           nrnb,
                        gmx_wallcycle_t   wcycle,
                        struct gmx_pme_t* pmedata,
                        int               variant,
                        rvec*             partgrad,
                        rvec*             MMgrad,
                        rvec*             MMgrad_full);

    real calculate_QMMM(const t_commrec*           cr,
                        gmx::ForceWithVirial*      forceWithVirial,
                              t_nrnb*              nrnb,
                              gmx_wallcycle_t      wcycle);

    // QMMM computes the QM forces.
    // This routine makes either function calls to gmx QM routines
    //   (derived from MOPAC7 (semi-emp.) and MPQC (ab initio))
    // or generates input files for an external QM package
    //   (listed in QMMMrec.QMpackage).
    // The binary of the QM package is called by system().
} ;

// for qmmm_variant
enum {eqmmmVACUO,
      eqmmmPME,
      eqmmmSWITCH,
      eqmmmRFIELD,
      eqmmmSHIFT,
      eqmmmNR};

/*! \brief
 * Return vector of atom indices for atoms in the QMMM region.
 *
 * \param[in] mtop Topology to use for populating array.
 * \param[in] ir   Inputrec used in simulation.
 * \returns Vector of atoms.
 */
std::vector<int> qmmmAtomIndices(const t_inputrec& ir,
                                 const gmx_mtop_t& mtop);

/*! \brief
 * Remove charges from QMMM atoms.
 *
 * \param[in] mtop Topology used for removing atoms.
 * \param[in] qmmmAtoms ArrayRef to vector conatining qmmm atom indices.
 */
void removeQmmmAtomCharges(gmx_mtop_t*              mtop,
                           gmx::ArrayRef<const int> qmmmAtoms);

#endif
