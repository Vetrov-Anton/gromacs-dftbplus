
#include "gmxpre.h"

#include "qmmm.h"

#include "config.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cmath>

#include <algorithm>

#include "gromacs/domdec/domdec_struct.h"
#include "gromacs/ewald/pme.h"
#include "gromacs/ewald/pme_internal.h"
#include "gromacs/fileio/confio.h"
#include "gromacs/gmxlib/network.h"
#include "gromacs/gmxlib/nrnb.h"
#include "gromacs/math/functions.h"
#include "gromacs/math/units.h"
#include "gromacs/math/vec.h"
#include "gromacs/mdlib/force.h"
#include "gromacs/mdtypes/commrec.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/mdtypes/mdatom.h"
#include "gromacs/mdtypes/nblist.h"
#include "gromacs/mdtypes/simulation_workload.h"
#include "gromacs/pbcutil/ishift.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/topology/mtop_lookup.h"
#include "gromacs/topology/mtop_util.h"
#include "gromacs/topology/topology.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/smalloc.h"

#define NM2BOHR           (1 / BOHR2NM)
#define NM_TO_BOHR        (18.897259886)
#define BOHR_TO_NM        (1 / NM_TO_BOHR)
#define HARTREE_TO_EV     (27.211396132)
// #define AU_OF_ESP_TO_VOLT (14.400) -- this is off by an angstrom/bohr factor! use HARTREE_TO_EV instead!
#define HARTREE2KJMOL     (HARTREE2KJ * AVOGADRO)
#define KJMOL2HARTREE     (1 / HARTREE2KJMOL)

#define QMMM_SWITCH       (0.05) // length of the additional switching region beyond cutoff

#define SQR(x) ((x)*(x))
#define CUB(x) ((x)*(x)*(x))
#define QRT(x) ((x)*(x)*(x)*(x))
#define QUI(x) ((x)*(x)*(x)*(x)*(x))
#define HEX(x) ((x)*(x)*(x)*(x)*(x)*(x))
#define OCT(x) ((x)*(x)*(x)*(x)*(x)*(x)*(x)*(x))
#define CHOOSE2(x) ((x)*((x)+1)/2)

#define gmx_erfc(x) (std::erfc(x))
#define gmx_erf(x)  (std::erf(x))

#include<time.h>

void print_time_difference(const char s[],
                           struct timespec start,
                           struct timespec end);

void print_time_difference(const char s[],
                           struct timespec start,
                           struct timespec end)
{
  //int sec, nsec;
  long long value = 1000000000ll * (static_cast<long long>(end.tv_sec)
                                  - static_cast<long long>(start.tv_sec))
                    + static_cast<long long>(end.tv_nsec - start.tv_nsec);
  printf("%s %12lld\n", s, value);
}  

/****************************************
 ******   AUXILIARY ROUTINES   **********
 ****************************************/

// adopted from src/gmxlib/pbc.c
static inline void pbc_dx_qmmm(matrix box, const rvec x1, const rvec x2, rvec dx)
{
    int i;

    for(i=0; i<DIM; i++) {
        dx[i] = x1[i] - x2[i];
        if (box != nullptr)
        {
            real length = box[i][i];
            while (dx[i] > length / 2.) {
                dx[i] -= length;
            }
            while (dx[i] < - length / 2.) {
                dx[i] += length;
            }
        }
    }
}

static inline real pbc_dist_qmmm(matrix box, const rvec x1, const rvec x2)
{
    rvec dx;

    for(int i=0; i<DIM; i++) {
        dx[i] = x1[i] - x2[i];

        if (box != nullptr)
        {
            real length = box[i][i];
            while (dx[i] > length / 2.) {
                dx[i] -= length;
            }
            while (dx[i] < - length / 2.) {
                dx[i] += length;
            }
        }
    }
    return norm(dx);
}

/****************************************
 ******   PRE-SCC CALCULATION  **********
 ****************************************/

void QMMM_rec::calculate_SR_QM_MM(int variant,
                                  real *pot,
                                  bool gradientRules,
                                  const real* mmCharges)
{
  /* as the last argument is expected: fr->ewaldcoeff_q */
  QMMM_QMrec& qm_ = qm[0];
  QMMM_MMrec& mm_ = mm[0];
  real rcoul = qm_.rcoulomb;
  real ewaldcoeff_q = qm_.ewaldcoeff_q;
  // The MM charges seen by the QM atoms: with GMX_QMMM_POT_SCHEME=AMBER including the charge
  //   of the MM1 atoms spread over the other MM atoms of their molecule (prepared with the
  //   short-range list); the MM1 atoms themselves are zeroed by qmmmScaleFactorPot()
  const real* qPot = potChargesSR.empty() ? mm_.MMcharges.data() : potChargesSR.data();
  /* energy_correction() asks for the same potential built with the rules of the gradient:
   *   the charges that the gradient uses (the link atoms of GMX_QMMM_GRAD_LA=exclude are then
   *   already spread over the MM atoms) and the factors of GMX_QMMM_GRAD_EXCL; the fictitious
   *   point charges of the boundary scheme belong to the Hamiltonian only and are left out. */
  if (gradientRules)
  {
      qPot = gradChargesMM.empty() ? mm_.MMcharges.data() : gradChargesMM.data();
  }
  /* receiver_potential() asks for the potential of a charge set of its own, with the
   * exclusions of the gradient. */
  if (mmCharges != nullptr)
  {
      qPot = mmCharges;
  }
  const auto scaleOf = [this, gradientRules](int j, int k) {
      return gradientRules ? mm[0].qmmmScaleFactorGrad(j, k) : mm[0].qmmmScaleFactorPot(j, k);
  };

  switch (variant) {

  case eqmmmVACUO: // no QM/MM
  {
      for (int j=0; j<qm_.nrQMatoms; j++)
      {
          pot[j] = 0.;
      }
      break;
  }

  case eqmmmSWITCH: // QM/MM with switched cut-off
  {
    real r_1 = rcoul;
    real r_d = QMMM_SWITCH;
    real r_c = r_1 + r_d;
  //printf("r_1 = %f, r_d = %f, r_c = %f\n", r_1, r_d, r_c);
    real big_a =   (5 * r_c - 2 * r_1) / (CUB(r_c) * SQR(r_d));
    real big_b = - (4 * r_c - 2 * r_1) / (CUB(r_c) * CUB(r_d));
    real big_c = - 1 / r_c - big_a / 3 * CUB(r_d) - big_b / 4 * QRT(r_d);
    for (int j=0; j<qm_.nrQMatoms; j++) { // do it for every QM atom
      pot[j] = 0.;
      int under_r1=0, under_rc=0;
      // add potential from MM atoms
      for (int k=0; k<mm_.nrMMatoms; k++) {
        // charge zeroed if this is an MM1 atom of the boundary charge scheme
        const real qMM = qPot[k] * scaleOf(j, k);
        if (qMM == 0.) {
          continue;
        }
        real r = pbc_dist_qmmm(qm_.box, qm_.xQM[j], mm_.xMM[k]);
        if (r < 0.001) { // this may occur on the first step of simulation for link atom(s)
          printf("QM--MM exploding for QM=%d, MM=%d. MM charge is %f\n", j+1, k+1, mm_.MMcharges[k]);
          continue;
        }
        if (r < r_1) {
          pot[j] += qMM * (1. / r + big_c);
          under_r1++;
          continue;
        }
        if (r < r_c) {
          pot[j] += qMM * ( 1. / r + big_a / 3. * CUB(r - r_1) + big_b / 4. * QRT(r - r_1) + big_c);
          under_rc++;
          continue;
        }
      } // for k
    //printf("ATOM %d: %8.5f %4d %4d\n", j+1, pot[j], under_r1, under_rc);
    } // for j
    break;
  }

  case eqmmmRFIELD: // reaction field with epsilon=infinity
  {
    real r_c = rcoul;
    real big_c = 3. / (2. * r_c);
    for (int j=0; j<qm_.nrQMatoms; j++) { // do it for every QM atom
      pot[j] = 0.;
      // add potential from MM atoms
      for (int k=0; k<mm_.nrMMatoms; k++) {
        // charge zeroed if this is an MM1 atom of the boundary charge scheme
        const real qMM = qPot[k] * scaleOf(j, k);
        if (qMM == 0.) {
          continue;
        }
        real r = pbc_dist_qmmm(qm_.box, qm_.xQM[j], mm_.xMM[k]);
        if (r < 0.001) { // this may occur on the first step of simulation for link atom(s)
          printf("QM--MM exploding for QM=%d, MM=%d. MM charge is %f\n", j+1, k+1, mm_.MMcharges[k]);
          continue;
        }
        if (r < r_c) {
          pot[j] += qMM * ( 1. / r + SQR(r) / 2. / CUB(r_c) - big_c);
          continue;
        }
      } // for k
    } // for j
    break;
  }

  case eqmmmSHIFT: // shifted cut-off, in a similar spirit as reaction field with epsilon=infinity
  {
    real r_c = rcoul;
    real big_c = 3. / SQR(r_c);
    real big_k = 3. / r_c;
    for (int j=0; j<qm_.nrQMatoms; j++) { // do it for every QM atom
      pot[j] = 0.;
      // add potential from MM atoms
      for (int k=0; k<mm_.nrMMatoms; k++) {
        // charge zeroed if this is an MM1 atom of the boundary charge scheme
        const real qMM = qPot[k] * scaleOf(j, k);
        if (qMM == 0.) {
          continue;
        }
        real r = pbc_dist_qmmm(qm_.box, qm_.xQM[j], mm_.xMM[k]);
        if (r < 0.001) { // this may occur on the first step of simulation for link atom(s)
          printf("QM--MM exploding for QM=%d, MM=%d. MM charge is %f\n", j+1, k+1, mm_.MMcharges[k]);
          continue;
        }
        if (r < r_c) {
          pot[j] += qMM * ( 1. / r - SQR(r) / CUB(r_c) + big_c * r - big_k);
          continue;
        }
      } // for k
    } // for j
    break;
  }

  case eqmmmPME: // QM/MM PME preparation -- short-range QM--MM component (real-space)
  {
    // using fr->ewaldcoeff_q, which has the dimension of 1/distance
    for (int j=0; j<qm_.nrQMatoms; j++) { // do it for every QM atom
      pot[j] = 0.;
      // add potential from MM atoms
      for (int k=0; k<mm_.nrMMatoms; k++) {
        /* With PME, a QM--MM pair zeroed by the boundary charge scheme needs two things:
         *   - its real-space term erfc(beta*r)/r is scaled with s, as for the cut-off variants;
         *   - the fraction (1-s) of its reciprocal-space term, which is computed on the grid
         *     over the full MM list in calculate_LR_QM_MM() and cannot be scaled there,
         *     is subtracted here as the pair term erf(beta*r)/r.
         * The sum of the two reproduces s/r for the pair, as it should.
         */
        const real s = scaleOf(j, k);
        const real qk = qPot[k]; // the same charge as on the grid of calculate_LR_QM_MM()
        real r = pbc_dist_qmmm(qm_.box, qm_.xQM[j], mm_.xMM[k]);
        if (r < 0.001) { // this may occur on the first step of simulation for link atom(s)
          printf("QM/MM PME QM--MM short range exploding for QM=%d, MM=%d. MM charge is %f\n", j+1, k+1, mm_.MMcharges[k]);
          if (s != 1.) { // erf(beta*r)/r is regular at r -> 0, so the correction still applies
            pot[j] -= (1. - s) * qk * M_2_SQRTPI * ewaldcoeff_q;
          }
        } else {
          /* The same cut-off as the real-space term of gradient_QM_MM(): the short-range
           * list holds every MM atom within rcoulomb of *any* QM atom, so a pair can be
           * farther than that, and the gradient drops it. The counter-term below is not
           * cut off, because the reciprocal-space contribution is not either.
           */
          if (r < rcoul) {
            pot[j] += s * qk / r * gmx_erfc(ewaldcoeff_q * r);
          }
          if (s != 1.) {
            pot[j] -= (1. - s) * qk / r * gmx_erf(ewaldcoeff_q * r);
          }
        }
      } // for k
    } // for j
    break;
  }

  default: // it should never get this far
    ;
  } // switch variant

  /* The fictitious charges of the boundary charge scheme, if any. */
  if (variant != eqmmmVACUO && !gradientRules)
  {
      add_boundary_scheme_potential(variant, pot);
  }

  /* Convert the result to atomic units. */
  for (int j=0; j<qm_.nrQMatoms; j++)
  {
      pot[j] /= NM2BOHR;
   // printf("SR QM/MM POT in a.u.: %d %8.5f\n", j+1, pot[j]);
  }
} // calculate_SR_QM_MM

/* Potential of the fictitious point charges of the boundary charge scheme
 *   (GMX_QMMM_POT_SCHEME = RC, RCD or CS) on the QM atoms, in e/nm.
 * The points exist for the QM atoms only: they are not on the PME grid and
 *   do not enter the gradient. With PME they act with the full 1/r, since they
 *   have no reciprocal-space part; with the cut-off variants they act with the
 *   same kernel as the MM atoms. Their positions follow the MM1 and MM2 atoms.
 */
void QMMM_rec::add_boundary_scheme_potential(int variant, real *pot)
{
  if (potPoints.empty())
  {
      return;
  }
  QMMM_QMrec& qm_ = qm[0];
  QMMM_MMrec& mm_ = mm[0];
  const real rcoul = qm_.rcoulomb;

  // the potential of a unit charge at distance r, as for the MM atoms of this variant
  const auto kernel = [variant, rcoul](real r) -> real {
      switch (variant)
      {
          case eqmmmSWITCH:
          {
              const real r_1   = rcoul;
              const real r_d   = QMMM_SWITCH;
              const real r_c   = r_1 + r_d;
              const real big_a = (5 * r_c - 2 * r_1) / (CUB(r_c) * SQR(r_d));
              const real big_b = -(4 * r_c - 2 * r_1) / (CUB(r_c) * CUB(r_d));
              const real big_c = -1 / r_c - big_a / 3 * CUB(r_d) - big_b / 4 * QRT(r_d);
              if (r < r_1)
              {
                  return 1. / r + big_c;
              }
              if (r < r_c)
              {
                  return 1. / r + big_a / 3. * CUB(r - r_1) + big_b / 4. * QRT(r - r_1) + big_c;
              }
              return 0.;
          }
          case eqmmmRFIELD:
              return r < rcoul ? 1. / r + SQR(r) / 2. / CUB(rcoul) - 3. / (2. * rcoul) : 0.;
          case eqmmmSHIFT:
              return r < rcoul ? 1. / r - SQR(r) / CUB(rcoul) + 3. / SQR(rcoul) * r - 3. / rcoul : 0.;
          case eqmmmPME:
              return 1. / r;
          default:
              return 0.;
      }
  };

  for (const PotPoint& p : potPoints)
  {
      const int ka = mm_.localIndexOfAtom[p.a];
      const int kb = mm_.localIndexOfAtom[p.b];
      if (ka < 0 || kb < 0)
      {
          gmx_fatal(FARGS,
                    "QM/MM boundary charge scheme: atom %d or %d is not on the short-range MM list "
                    "of the QM atoms. Increase rcoulomb.",
                    p.a + 1, p.b + 1);
      }
      rvec ab, x;
      pbc_dx_qmmm(qm_.box, mm_.xMM[kb], mm_.xMM[ka], ab); // x(b) - x(a), nearest image
      for (int d = 0; d < DIM; d++)
      {
          x[d] = mm_.xMM[ka][d] + p.f * ab[d];
      }
      const real q = p.q * mm_.scalefactor;
      for (int j = 0; j < qm_.nrQMatoms; j++)
      {
          const real r = pbc_dist_qmmm(qm_.box, qm_.xQM[j], x);
          if (r < 0.001)
          {
              printf("QM/MM boundary charge exploding for QM=%d near MM atoms %d, %d\n", j + 1,
                     p.a + 1, p.b + 1);
              continue;
          }
          pot[j] += q * kernel(r);
      }
  }
} // add_boundary_scheme_potential

/* The electrostatic gradient of those same fictitious point charges, in hartree/bohr.
 *   Each point sits at x = (1 - f) * x(MM1) + f * x(MM2) and carries no mass of its own, so
 *   the force on it is passed to MM1 and MM2 by the chain rule, with the weights (1 - f)
 *   and f. The kernel is the derivative of the one in add_boundary_scheme_potential(), so
 *   that this gradient belongs to that potential exactly; with PME the points act with the
 *   full 1/r in both, as they are not on the grid.
 * Only the gradient with the rules of the potential uses this -- the gradient that mdrun
 *   applies to the atoms never contains the fictitious charges.
 */
void QMMM_rec::add_boundary_scheme_gradient(int variant, const real* qQM, rvec* partgrad, rvec* MMgrad)
{
  if (potPoints.empty())
  {
      return;
  }
  QMMM_QMrec& qm_ = qm[0];
  QMMM_MMrec& mm_ = mm[0];
  const real rcoul = qm_.rcoulomb;

  // d/dr of the kernel of add_boundary_scheme_potential(), for the same variant
  const auto dkernel = [variant, rcoul](real r) -> real {
      switch (variant)
      {
          case eqmmmSWITCH:
          {
              const real r_1   = rcoul;
              const real r_d   = QMMM_SWITCH;
              const real r_c   = r_1 + r_d;
              const real big_a = (5 * r_c - 2 * r_1) / (CUB(r_c) * SQR(r_d));
              const real big_b = -(4 * r_c - 2 * r_1) / (CUB(r_c) * CUB(r_d));
              if (r < r_1)
              {
                  return -1. / SQR(r);
              }
              if (r < r_c)
              {
                  return -1. / SQR(r) + big_a * SQR(r - r_1) + big_b * CUB(r - r_1);
              }
              return 0.;
          }
          case eqmmmRFIELD:
              return r < rcoul ? -1. / SQR(r) + r / CUB(rcoul) : 0.;
          case eqmmmSHIFT:
              return r < rcoul ? -1. / SQR(r) - 2. * r / CUB(rcoul) + 3. / SQR(rcoul) : 0.;
          case eqmmmPME:
              return -1. / SQR(r);
          default:
              return 0.;
      }
  };

  for (const PotPoint& p : potPoints)
  {
      const int ka = mm_.localIndexOfAtom[p.a];
      const int kb = mm_.localIndexOfAtom[p.b];
      if (ka < 0 || kb < 0)
      {
          gmx_fatal(FARGS,
                    "QM/MM boundary charge scheme: atom %d or %d is not on the short-range MM list "
                    "of the QM atoms. Increase rcoulomb.",
                    p.a + 1, p.b + 1);
      }
      rvec ab, x;
      pbc_dx_qmmm(qm_.box, mm_.xMM[kb], mm_.xMM[ka], ab); // x(b) - x(a), nearest image
      for (int d = 0; d < DIM; d++)
      {
          x[d] = mm_.xMM[ka][d] + p.f * ab[d];
      }
      const real q = p.q * mm_.scalefactor;
      for (int j = 0; j < qm_.nrQMatoms; j++)
      {
          rvec bond;
          pbc_dx_qmmm(qm_.box, qm_.xQM[j], x, bond); // x(QM j) - x(point)
          const real r = norm(bond);
          if (r < 0.001)
          {
              printf("QM/MM boundary charge exploding for QM=%d near MM atoms %d, %d\n", j + 1,
                     p.a + 1, p.b + 1);
              continue;
          }
          const real dk = dkernel(r);
          if (dk == 0.)
          {
              continue;
          }
          rvec dgr;
          svmul(qQM[j] * q * dk / r * SQR(BOHR2NM), bond, dgr);
          rvec_inc(partgrad[j], dgr);
          // the point has no coordinates of its own: it follows MM1 and MM2
          for (int d = 0; d < DIM; d++)
          {
              MMgrad[ka][d] -= (1. - p.f) * dgr[d];
              MMgrad[kb][d] -= p.f * dgr[d];
          }
      }
  }
} // add_boundary_scheme_gradient

/* Calculate the effect of environment with PME,
 * EXCLUDING the periodic images of QM charges at this stage!
 * That contribution will be added within the SCC cycle.
 */
void QMMM_rec::calculate_LR_QM_MM(const t_commrec *cr,
                                  t_nrnb *nrnb,
                                  gmx_wallcycle_t wcycle,
                                  struct gmx_pme_t *pmedata,
                                  real *pot,
                                  bool gradientRules,
                                  const real* mmCharges)
{
  QMMM_QMrec& qm_      = qm[0];
  QMMM_MMrec& mm_      = mm[0];
  QMMM_PME& pme_full   = pme[0];
  QMMM_PME& pme_qmonly = pme[1];
//QMMM_PME *pme_full   = qr->pme_full;
//QMMM_PME *pme_qmonly = qr->pme_qmonly;
  const int   n  = qm_.nrQMatoms;
  const int   ne = mm_.nrMMatoms_full;
//const int   ntot = n + ne;
  // GMX_QMMM_POT_SCHEME=AMBER: the MM1 charges spread over the MM atoms, see calculate_SR_QM_MM();
  //   the shifts on the full list are prepared once, the list being static
  if (!potChargeShift.empty() && potShiftFull.empty())
  {
      potShiftFull.resize(ne);
      for (int j=0; j<ne; j++)
      {
          potShiftFull[j] = potChargeShiftOf(mm_.indexMM_full[j]) * mm_.scalefactor;
      }
  }

  /* copy the data into PME structures */
  for (int j=0; j<n; j++)
  {
      /* QM atoms -- no charges! */
      pme_full.x[j][XX] = qm_.xQM[j][XX];
      pme_full.x[j][YY] = qm_.xQM[j][YY];
      pme_full.x[j][ZZ] = qm_.xQM[j][ZZ];
      pme_full.q[j]     = 0.;
   // printf("QM %5d %8.5f %8.5f %8.5f\n", j+1, pme_full.x[j][XX], pme_full.x[j][YY], pme_full.x[j][ZZ]);

      pme_qmonly.x[j][XX] = qm_.xQM[j][XX];
      pme_qmonly.x[j][YY] = qm_.xQM[j][YY];
      pme_qmonly.x[j][ZZ] = qm_.xQM[j][ZZ];
      pme_qmonly.q[j]     = 0.;
  }
  for (int j=0; j<ne; j++)
  {
      /* MM atoms -- with charges */
      pme_full.x[n + j][XX] = mm_.xMM_full[j][XX];
      pme_full.x[n + j][YY] = mm_.xMM_full[j][YY];
      pme_full.x[n + j][ZZ] = mm_.xMM_full[j][ZZ];
      pme_full.q[n + j]     = (mmCharges != nullptr)
                                      ? mmCharges[j]
                                      : (gradientRules ? gradChargeMMfull(j) : mm_.MMcharges_full[j]);
   // printf("MM %5d %8.5f %8.5f %8.5f %8.5f\n", j+1, pme->x[n+j][XX], pme->x[n+j][YY], pme->x[n+j][ZZ], pme->q[n + j]);
  }
  /* the AMBER shift belongs to the Hamiltonian only, so the rules of the gradient skip it */
  if (!potShiftFull.empty() && !gradientRules && mmCharges == nullptr)
  {
      for (int j=0; j<ne; j++)
      {
          pme_full.q[n + j] += potShiftFull[j];
      }
  }
  
//static struct timespec time1, time2;
//clock_gettime(CLOCK_MONOTONIC, &time1);
  // init_nrnb(pme_full.nrnb); // TODO change to something?
  gmx::StepWorkload stepWork;
  stepWork.computePotentials = true;
  PaddedVector<gmx::RVec> emptyVec;
  gmx_pme_do(pmedata, gmx::makeArrayRef(pme_full.x), gmx::makeArrayRef(emptyVec), pme_full.q.data(), pme_full.q.data(),
             nullptr, nullptr, nullptr, nullptr, qm_.box, cr, 0, 0, //pme_full.nrnb->get(),
             nrnb, wcycle, pme_full.vir, pme_full.vir, nullptr, nullptr, 0., 0., nullptr, nullptr,
             stepWork, TRUE, FALSE, n, pme_full.pot);
//clock_gettime(CLOCK_MONOTONIC, &time2);
//print_time_difference("PMETIME 1 ", time1, time2);

  /* Save the potential */
  for (int j=0; j<qm_.nrQMatoms; j++)
  {
      pot[j] = pme_full.pot[j] * KJMOL2HARTREE; // conversion OK
  }

  if (pme_full.surf_corr_pme)
  {
      /* optionally evaluate the PME surface correction term.
       * ATTENTION: modified update_QMMM_coord() (qmmm.cpp) is needed here!
       */
       // sum_j q_j vec(x_j)
       rvec qx, sum_qx;
       clear_rvec(sum_qx);
    // rvec subsum_qx;
	   for (int j=0; j<ne; j++) {
           svmul(pme_full.q[n + j], mm_.xMM_full[j], qx);
           rvec_inc(sum_qx, qx);
        // if (j%3==0) {
        //   printf("MOL %4d DIPOLE %5.1f %5.1f %5.1f\n", j/3,
        //     subsum_qx[XX]/0.020819434, subsum_qx[YY]/0.020819434, subsum_qx[ZZ]/0.020819434);
        //   clear_rvec(subsum_qx);
        // }
        // rvec_inc(subsum_qx, qx);
	   }
	   // contribution to the potential
       std::vector<real> pot_add(n);
	   real vol = qm_.box[XX][XX] * qm_.box[YY][YY] * qm_.box[ZZ][ZZ];
	   for (int j=0; j<n; j++) {
	       pot_add[j] = 4. * M_PI / 3. / vol / pme_full.epsilon_r * iprod(qm_.xQM[j], sum_qx) / NM2BOHR;
       }
    // printf("VOL = %8.5f, EPS_R = %8.5f, DIP = %9.5f %9.5f %9.5f\n",
    //         vol, pme_full.epsilon_r, sum_qx[XX], sum_qx[YY], sum_qx[ZZ]);
	   for (int j=0; j<n; j++) {
    //     printf("POT LR [%d] = %8.5f + %8.5f\n", j+1, pot[j], pot_add[j]);
           pot[j] += pot_add[j];
       }
  }
  else
  {
	// for (int j=0; j<n; j++) {
    //     printf("POT LR [%d] = %8.5f\n", j+1, pot[j]);
    // }
  }

//for (int j=0; j<n; j++) {
//    printf("POT LR [%d] = %8.5f\n", j+1, pot[j]);
//}
} // calculate_LR_QM_MM

/****************************************
 *********  IN-SCC CALCULATION  *********
 ****************************************/

/* Calculate the potential induced by the periodic images of QM charges.
 * This needs to be performed in every SCC iteration.
 */
/*
void calculate_complete_QM_QM_ewald(t_QMMMrec *qr,
                              const t_commrec *cr,
                              gmx_wallcycle_t wcycle,
                              struct gmx_pme_t *pmedata,
                              real *pot)
{
  // as the last arguments are expected: fr->ewaldcoeff_q and fr->pmedata
  t_QMrec    *qm           = qr->qm[0];
  t_QMMM_PME *pme          = qr->pme;
  int         n            = qm->nrQMatoms;
//real        rcoul        = qm->rcoulomb;
  real        ewaldcoeff_q = qm->ewaldcoeff_q;

  // copy the data into PME structures
  for (int j=0; j<n; j++)
  {
      // QM atoms with charges
      pme->x[j][XX] = qm->xQM[j][XX];
      pme->x[j][YY] = qm->xQM[j][YY];
      pme->x[j][ZZ] = qm->xQM[j][ZZ];
      // Attenuate the periodic images of the QM zone
      // with the same scaling factor
      // that is applied for the MM atoms.
      pme->q[j]     = qm->QMcharges[j] * qr->mm->scalefactor;
  }
  
  static struct timespec time1, time2;
  clock_gettime(CLOCK_MONOTONIC, &time1);
  init_nrnb(pme->nrnb);

  for (int j=0; j<n; j++) {
    pme->pot[j] = 0.;
  }

  rvec kvec;
  for (int kxi=-pmedata->nkx; kxi<=pmedata->nkx; kxi++) {
    kvec[XX] = (real) kxi * 2. * M_PI / qm->box[XX][XX];
    for (int kyi=-pmedata->nky; kyi<=pmedata->nky; kyi++) {
      kvec[YY] = (real) kyi * 2. * M_PI / qm->box[YY][YY];
      for (int kzi=-pmedata->nkz; kzi<=pmedata->nkz; kzi++) {
        kvec[ZZ] = (real) kzi * 2. * M_PI / qm->box[ZZ][ZZ];

        if (kxi != 0 || kyi != 0 || kzi != 0) {
          real factor = exp(-norm2(kvec) / 4. / SQR(pmedata->ewaldcoeff_q)) / norm2(kvec);
        //printf("KX %2d KY %2d KZ %2d FACTOR %12.7f\n", kxi, kyi, kzi, factor);
          for (int j=0; j<n; j++) {
            for (int k=0; k<n; k++) {
              rvec bond;
              rvec_sub(pme->x[k], pme->x[j], bond);
              real addend = factor * pme->q[k] * cos((double) iprod(kvec, bond));
              pme->pot[j] += addend;
            //printf("J %2d K %2d X %8.5f Y %8.5f Z %8.5f ADDEND %12.7f\n",
            //  j, k, bond[XX], bond[YY], bond[ZZ], addend);
            }
          }
        }

      }
    }
  }

  for (int j=0; j<n; j++) {
    real factor = 4. * M_PI / (qm->box[XX][XX] * qm->box[YY][YY] * qm->box[ZZ][ZZ])
                * ONE_4PI_EPS0 / pmedata->epsilon_r;
  //printf("final factor = %12.7f\n", factor);
    pme->pot[j] *= factor;
  }

  printf("NKX %d NKY %d NKZ %d\n", pmedata->nkx, pmedata->nky, pmedata->nkz);
  for (int j=0; j<n; j++) {
    printf("POT QM QM [%3d] = %12.7f\n", j, pme->pot[j]);
  }

  clock_gettime(CLOCK_MONOTONIC, &time2);
  print_time_difference("EWATIME 2 ", time1, time2);
  
  // short-range corrections
  std::vector<real> pot_corr(n);
  for (int j=0; j<n; j++)
  {
      // exclude the interaction of atom j with its own charge density
      pot_corr[j] = - 2. * ewaldcoeff_q * pme->q[j] / sqrt(M_PI);
      // exclude the interactions with the other QM atoms
      for (int k=0; k<n; k++)
      {
          if (j != k)
          {
              real r = pbc_dist_qmmm(nullptr, pme->x[j], pme->x[k]);
              pot_corr[j] -= pme->q[k] * gmx_erf(ewaldcoeff_q * r) / r;
          }
      }
  }
      
  std::vector<real> pot_surf(n);
  if (pme->surf_corr_pme)
  {
      // optionally evaluate the PME surface correction term.
      // ATTENTION: modified update_QMMM_coord() (qmmm.c) is needed here!
       // sum_j q_j vec(x_j)
       rvec qx, sum_qx;
       clear_rvec(sum_qx);
	   for (int j=0; j<qm->nrQMatoms; j++) {
           svmul(pme->q[j], pme->x[j], qx);
           rvec_inc(sum_qx, qx);
	   }
	   // contribution to the potential
	   real vol = qm->box[XX][XX] * qm->box[YY][YY] * qm->box[ZZ][ZZ];
	   for (int j=0; j<n; j++) {
	       pot_surf[j] = 4. * M_PI / 3. / vol / pme->epsilon_r * iprod(qm->xQM[j], sum_qx);
       }
  }
  else
  {
	   for (int j=0; j<n; j++) {
           pot_surf[j] = 0.;
	   }
  }

  // return the potential on QM atoms
  for (int j=0; j<n; j++)
  {
      pot[j] = pme->pot[j] * KJMOL2HARTREE + pot_corr[j] * BOHR2NM + pot_surf[j] * BOHR2NM;
   // printf("pot_qm_in_scc[%d] = %12.8f\n", j+1, pot[j]);
   // printf("Ewald atom %d charge %6.3f potential %8.5f (correction %8.5f surfterm %8.5f)\n",
   //         j+1, pme->q[j],      pot[j],                 pot_corr[j] * BOHR2NM, pot_surf[j] * BOHR2NM);
  }

  return;
} // calculate_complete_QM_QM_ewald
*/

void QMMM_rec::calculate_complete_QM_QM(const t_commrec*  cr,
                                        t_nrnb*           nrnb,
                                        gmx_wallcycle_t   wcycle,
                                        struct gmx_pme_t* pmedata,
                                        real*             pot,
                                        const real*       charges)
{
  /* as the last arguments are expected: fr->ewaldcoeff_q and fr->pmedata */
  QMMM_QMrec& qm_          = qm[0];
  QMMM_MMrec& mm_          = mm[0];
//QMMM_PME& pme_full       = pme[0];
  QMMM_PME& pme_qmonly     = pme[1];
//t_QMMM_PME *pme          = qr->pme_qmonly;
  int         n            = qm_.nrQMatoms;
//real        rcoul        = qm_.rcoulomb;
  real        ewaldcoeff_q = qm_.ewaldcoeff_q;

///* (re)allocate */
//srenew(pme->x, qm_.nrQMatoms);
//srenew(pme->q, qm_.nrQMatoms);

  /* copy the data into PME structures */
  for (int j=0; j<n; j++)
  {
      /* QM atoms with charges */
      pme_qmonly.x[j][XX] = qm_.xQM[j][XX];
      pme_qmonly.x[j][YY] = qm_.xQM[j][YY];
      pme_qmonly.x[j][ZZ] = qm_.xQM[j][ZZ];
      /* Attenuate the periodic images of the QM zone
       * with the same scaling factor
       * that is applied for the MM atoms.
       */
      pme_qmonly.q[j]     = (charges == nullptr ? qm_.QMcharges[j] : charges[j]) * mm_.scalefactor;
  }
  
//static struct timespec time1, time2;
//clock_gettime(CLOCK_MONOTONIC, &time1);
  // init_nrnb(pme_qmonly.nrnb); // TODO change to something?
  gmx::StepWorkload stepWork;
  stepWork.computePotentials = true;
  PaddedVector<gmx::RVec> emptyVec;
  int oldNumAtoms = pmedata->atc[0].numAtoms(); // need to resize PME arrays to the number of QM atoms
  pmedata->atc[0].setNumAtoms(n);
  gmx_pme_do(pmedata, pme_qmonly.x, gmx::makeArrayRef(emptyVec), pme_qmonly.q.data(), pme_qmonly.q.data(),
             nullptr, nullptr, nullptr, nullptr, qm_.box, cr, 0, 0, // pme_qmonly.nrnb->get(),
             nrnb, wcycle, pme_qmonly.vir, pme_qmonly.vir, nullptr, nullptr, 0., 0., nullptr, nullptr,
             stepWork, TRUE, FALSE, n, pme_qmonly.pot);
  pmedata->atc[0].setNumAtoms(oldNumAtoms); // resize back
//clock_gettime(CLOCK_MONOTONIC, &time2);
//print_time_difference("PMETIME 2 ", time1, time2);

//for (int j=0; j<n; j++) {
//  printf("POT QM QM [%3d] = %12.7f\n", j, pme_qmonly.pot[j]);
//}
  
  /* short-range corrections */
  std::vector<real> pot_corr(n);
  for (int j=0; j<n; j++)
  {
      /* exclude the interaction of atom j with its own charge density */
      pot_corr[j] = - 2. * ewaldcoeff_q * pme_qmonly.q[j] / sqrt(M_PI);
      /* exclude the interactions with the other QM atoms */
      for (int k=0; k<n; k++)
      {
          if (j != k)
          {
              real r = pbc_dist_qmmm(nullptr, pme_qmonly.x[j], pme_qmonly.x[k]);
              pot_corr[j] -= pme_qmonly.q[k] * gmx_erf(ewaldcoeff_q * r) / r;
          }
      }
  }
      
  std::vector<real> pot_surf(n);
  if (pme_qmonly.surf_corr_pme)
  {
      /* optionally evaluate the PME surface correction term.
       * ATTENTION: modified update_QMMM_coord() (qmmm.c) is needed here!
       */
       // sum_j q_j vec(x_j)
       rvec qx, sum_qx;
       clear_rvec(sum_qx);
	   for (int j=0; j<qm_.nrQMatoms; j++) {
           svmul(pme_qmonly.q[j], pme_qmonly.x[j], qx);
           rvec_inc(sum_qx, qx);
	   }
	   // contribution to the potential
	   real vol = qm_.box[XX][XX] * qm_.box[YY][YY] * qm_.box[ZZ][ZZ];
	   for (int j=0; j<n; j++) {
	       pot_surf[j] = 4. * M_PI / 3. / vol / pme_qmonly.epsilon_r * iprod(qm_.xQM[j], sum_qx);
       }
  }
  else
  {
	   for (int j=0; j<n; j++) {
           pot_surf[j] = 0.;
	   }
  }

  /* return the potential on QM atoms */
  for (int j=0; j<n; j++)
  {
      pot[j] = pme_qmonly.pot[j] * KJMOL2HARTREE + pot_corr[j] * BOHR2NM + pot_surf[j] * BOHR2NM;
   // printf("pot_qm_in_scc[%d] = %12.8f\n", j+1, pot[j]);
   // printf("Ewald atom %d charge %6.3f potential %8.5f (correction %8.5f surfterm %8.5f)\n",
   //         j+1, pme_qmonly.q[j],      pot[j],                 pot_corr[j] * BOHR2NM, pot_surf[j] * BOHR2NM);
  }

  // also, save the potential in the QMMM_QMrec structure -- but not when this is an extra
  //   evaluation with another charge set, which must leave the record of the step untouched
  if (charges == nullptr)
  {
      for (int j=0; j<n; j++)
      {
          qm_.pot_qmqm_set(j, static_cast<double>(pot[j]) * HARTREE_TO_EV); // in volt units
      }
  }
} // calculate_complete_QM_QM

/**********************************
 ***  GRADIENTS       *************
 **********************************/

/* GMX_QMMM_GRAD_LA=exclude: the potential on the QM atoms of a unit charge on every MM atom
 *   that receives the charge of the link atoms of this molecule, built with the exclusions of
 *   the gradient. It is a purely geometric quantity -- it does not contain the Mulliken
 *   charges -- and it is what turns the response of a link atom into a single number, see
 *   energy_correction(). A receiver that is off the short-range list contributes through the
 *   reciprocal space only, which is right: its real-space term is beyond the cut-off.
 */
void QMMM_rec::receiver_potential(int               molecule,
                                  const t_commrec*  cr,
                                  t_nrnb*           nrnb,
                                  gmx_wallcycle_t   wcycle,
                                  struct gmx_pme_t* pmedata,
                                  int               variant,
                                  real*             pot)
{
    QMMM_QMrec& qm_ = qm[0];
    QMMM_MMrec& mm_ = mm[0];
    const int   n   = qm_.nrQMatoms;

    receiverChargesSR.assign(mm_.nrMMatoms, real(0.0));
    for (int a : gradLaMolecules[molecule].receivers)
    {
        const int k = mm_.localIndexOfAtom[a];
        if (k >= 0)
        {
            receiverChargesSR[k] = real(1.0);
        }
    }
    calculate_SR_QM_MM(variant, pot, true, receiverChargesSR.data());

    if (variant == eqmmmPME)
    {
        // the full MM list is static, so the reverse map is built once
        if (fullIndexOfAtom.empty())
        {
            fullIndexOfAtom.assign(mm_.localIndexOfAtom.size(), -1);
            for (int k = 0; k < mm_.nrMMatoms_full; k++)
            {
                fullIndexOfAtom[mm_.indexMM_full[k]] = k;
            }
        }
        receiverChargesFull.assign(mm_.nrMMatoms_full, real(0.0));
        for (int a : gradLaMolecules[molecule].receivers)
        {
            const int k = fullIndexOfAtom[a];
            if (k >= 0)
            {
                receiverChargesFull[k] = real(1.0);
            }
        }
        std::vector<real> potLr(n, real(0.0));
        calculate_LR_QM_MM(cr, nrnb, wcycle, pmedata, potLr.data(), true, receiverChargesFull.data());
        for (int j = 0; j < n; j++)
        {
            pot[j] += potLr[j];
        }
    }
} // receiver_potential

/* The energy of the QM--MM electrostatics under the rules of the gradient
 *   (GMX_QMMM_ENERGY_CORRECTION=on, the default).
 *
 * DFTB+ returns an energy that contains the interaction of the QM charges with the potential
 *   that went into its Hamiltonian, i.e. built with GMX_QMMM_POT_SCHEME:
 *
 *     E(DFTB+) = E_QM[q] + sum_A q_A * phi^pot_A ,
 *
 * whereas the forces of gradient_QM_MM() are the derivative, at frozen charges, of
 *
 *     sum_A q^grad_A * phi^grad_A ,
 *
 *   with the charges and the exclusions of GMX_QMMM_GRAD_EXCL, GMX_QMMM_GRAD_LA and
 *   GMX_QMMM_FUDGE_QQ. This routine returns the difference of the two, in hartree, so that
 *   the caller can replace one by the other and report an energy that belongs to the forces.
 *
 * With PME the potential of the periodic images of the QM charges is treated the same way:
 *   it enters the DFTB+ energy with the Mulliken charges and the gradient with the charges of
 *   the gradient. The two sets differ only with GMX_QMMM_GRAD_LA=exclude; when they are equal
 *   the image terms cancel exactly and no extra PME evaluation is done. The convention of that
 *   term (the factor of the image self-interaction) is not touched here, only the charge set.
 *
 * Nothing is corrected when the rules of the potential and of the gradient coincide: the two
 *   sums are then identical term by term, and the result is zero to round-off.
 */
double QMMM_rec::energy_correction(const t_commrec*     cr,
                                   t_nrnb*              nrnb,
                                   gmx_wallcycle_t      wcycle,
                                   struct gmx_pme_t*    pmedata,
                                   int                  variant,
                                   std::vector<double>* dVout)
{
    if (dVout != nullptr)
    {
        dVout->clear();
    }
    if (!energyCorrection || variant == eqmmmVACUO)
    {
        return 0.;
    }
    QMMM_QMrec& qm_ = qm[0];
    const int   n   = qm_.nrQMatoms;
    energyPotWork.resize(n);
    energyPotWorkLr.assign(n, real(0.0));

    // QM--MM with the rules of the gradient
    std::vector<double> phiGrad(n);
    calculate_SR_QM_MM(variant, energyPotWork.data(), true);
    if (variant == eqmmmPME)
    {
        calculate_LR_QM_MM(cr, nrnb, wcycle, pmedata, energyPotWorkLr.data(), true);
    }
    for (int j = 0; j < n; j++)
    {
        phiGrad[j] = static_cast<double>(energyPotWork[j]) + static_cast<double>(energyPotWorkLr[j]);
    }

    // QM--MM with the rules of the Hamiltonian, i.e. what DFTB+ has already counted
    std::vector<double> phiPot(n);
    energyPotWorkLr.assign(n, real(0.0));
    calculate_SR_QM_MM(variant, energyPotWork.data(), false);
    if (variant == eqmmmPME)
    {
        calculate_LR_QM_MM(cr, nrnb, wcycle, pmedata, energyPotWorkLr.data(), false);
    }
    for (int j = 0; j < n; j++)
    {
        phiPot[j] = static_cast<double>(energyPotWork[j]) + static_cast<double>(energyPotWorkLr[j]);
    }

    double eGrad = 0.;
    double ePot  = 0.;
    for (int j = 0; j < n; j++)
    {
        eGrad += static_cast<double>(gradChargeQM(j)) * phiGrad[j];
        ePot += static_cast<double>(qm_.QMcharges[j]) * phiPot[j];
    }

    /* Periodic images of the QM charges, only when the two charge sets differ -- otherwise
     * the two terms are identical and cancel. The factor of 1/2 is the one of the Ewald
     * energy of a charge distribution with its own images; call_dftbplus() has already
     * replaced the full term that DFTB+ counted by the halved one, so what is left here is
     * half of the difference between the two charge sets.
     */
    std::vector<double> imgGrad, imgPot;
    if (variant == eqmmmPME && !gradChargesQM.empty())
    {
        imgGrad.resize(n);
        imgPot.resize(n);
        calculate_complete_QM_QM(cr, nrnb, wcycle, pmedata, energyPotWork.data(), gradChargesQM.data());
        for (int j = 0; j < n; j++)
        {
            imgGrad[j] = static_cast<double>(energyPotWork[j]);
            eGrad += 0.5 * static_cast<double>(gradChargesQM[j]) * imgGrad[j];
        }
        calculate_complete_QM_QM(cr, nrnb, wcycle, pmedata, energyPotWork.data(), qm_.QMcharges);
        for (int j = 0; j < n; j++)
        {
            imgPot[j] = static_cast<double>(energyPotWork[j]);
            ePot += 0.5 * static_cast<double>(qm_.QMcharges[j]) * imgPot[j];
        }
    }

    /* The perturbation that the response correction of the forces applies to the SCC:
     *   dE/dq at the SCC solution. The charges are stationary in phi_pot plus the potential
     *   of their own images, so dG/dq = -(phi_pot + V_img[q]), and what is left is the
     *   derivative of the terms above.
     *
     * With the plain charge sets (every GMX_QMMM_GRAD_LA but exclude) the QM--MM term is
     *   linear in q and the image terms cancel, so this is simply phi_grad - phi_pot.
     *
     * With GMX_QMMM_GRAD_LA=exclude a link atom L carries no charge in the gradient, and
     *   its Mulliken charge is spread over the receivers of its molecule, q_k += c * q_L
     *   with c = scalefactor / (number of receivers). Both charge sets then depend on q,
     *   the QM--MM term is quadratic in it, and the derivative differs per atom:
     *
     *     dE/dq_A = phi_grad(A) - phi_pot(A) + V_img[q_grad](A) - V_img[q](A)   (A not a link atom)
     *     dE/dq_L = c * sum_A q_grad(A) * phi_unit(A) - phi_pot(L) - V_img[q](L)
     *
     *   where phi_unit is the potential of a unit charge on every receiver, from
     *   receiver_potential(). The sum over the receivers of the potential of the QM charges
     *   is rewritten that way with the symmetry of the interaction, so that one extra
     *   evaluation per molecule replaces a sum over all of the receivers.
     */
    if (dVout != nullptr)
    {
        dVout->assign(n, 0.);
        if (gradChargesQM.empty())
        {
            for (int j = 0; j < n; j++)
            {
                (*dVout)[j] = phiGrad[j] - phiPot[j];
            }
        }
        else
        {
            if (laMoleculeOfQmAtom.empty())
            {
                laMoleculeOfQmAtom.assign(n, -1);
                for (size_t m = 0; m < gradLaMolecules.size(); m++)
                {
                    for (int j : gradLaMolecules[m].linkAtomsOfQmList)
                    {
                        laMoleculeOfQmAtom[j] = static_cast<int>(m);
                    }
                }
            }
            std::vector<double> psi(gradLaMolecules.size(), 0.);
            std::vector<real>   phiUnit(n);
            for (size_t m = 0; m < gradLaMolecules.size(); m++)
            {
                receiver_potential(static_cast<int>(m), cr, nrnb, wcycle, pmedata, variant,
                                   phiUnit.data());
                double sum = 0.;
                for (int j = 0; j < n; j++)
                {
                    sum += static_cast<double>(gradChargeQM(j)) * static_cast<double>(phiUnit[j]);
                }
                psi[m] = sum * static_cast<double>(mm[0].scalefactor)
                         / static_cast<double>(gradLaMolecules[m].receivers.size());
            }
            for (int j = 0; j < n; j++)
            {
                const double vImgPot  = imgPot.empty() ? 0. : imgPot[j];
                const double vImgGrad = imgGrad.empty() ? 0. : imgGrad[j];
                const int    mol      = laMoleculeOfQmAtom[j];
                (*dVout)[j] = (mol < 0) ? (phiGrad[j] - phiPot[j]) + (vImgGrad - vImgPot)
                                        : psi[mol] - phiPot[j] - vImgPot;
            }
        }
    }

    return eGrad - ePot;
} // energy_correction

/* Periodic image of x nearest to the reference position ref (rectangular box, as
 * everywhere in this file). */
static void qmmmNearestImage(const matrix box, const rvec ref, const rvec x, rvec image)
{
    for (int d = 0; d < DIM; d++)
    {
        const real L  = box[d][d];
        real       dx = x[d] - ref[d];
        if (L > 0)
        {
            dx -= L * std::round(dx / L);
        }
        image[d] = ref[d] + dx;
    }
}

/* w += x (x) f */
static void qmmmAddOuter(matrix w, const rvec x, const rvec f)
{
    for (int a = 0; a < DIM; a++)
    {
        for (int b = 0; b < DIM; b++)
        {
            w[a][b] += x[a] * f[b];
        }
    }
}

void QMMM_rec::gradient_QM_MM(const t_commrec*  cr,
                              t_nrnb*           nrnb,
                              gmx_wallcycle_t   wcycle,
                              struct gmx_pme_t* pmedata,
                              int               variant,
                              rvec*             partgrad,
                              rvec*             MMgrad,
                              rvec*             MMgrad_full,
                              bool              potentialRules)
{
  QMMM_QMrec& qm_ = qm[0];
  QMMM_MMrec& mm_ = mm[0];
//t_QMMM_PME *pme_full   = qr->pme_full;
//t_QMMM_PME *pme_qmonly = qr->pme_qmonly;
  /* GMX_QMMM_GRAD_LA=exclude: this routine then works with the link atoms zeroed and
   * their charge spread over the MM atoms of their molecule -- in the QM--MM gradient
   * and in the gradient of the QM periodic images alike. With the other settings of
   * GMX_QMMM_GRAD_LA these are the plain Mulliken and force-field charges, and nothing
   * below changes. The external potential of the SCC calculation is never affected.
   *
   * With potentialRules the sources are those of the external potential instead
   * (GMX_QMMM_POT_SCHEME): the MM charges that polarize the QM density, with the MM1 atoms
   * zeroed by qmmmScaleFactorPot() and the fictitious point charges of the boundary scheme
   * added at the end, and the plain Mulliken charges on the QM atoms. The exclusions of the
   * gradient play no role then. This is what the response correction of the forces needs;
   * it is never the gradient that mdrun applies to the atoms.
   */
  if (!potentialRules)
  {
      update_gradient_charges(variant);
  }
  const real* qQM = potentialRules
                            ? qm_.QMcharges
                            : (gradChargesQM.empty() ? qm_.QMcharges : gradChargesQM.data());
  const real* qMMsr =
          potentialRules ? (potChargesSR.empty() ? mm_.MMcharges.data() : potChargesSR.data())
                         : (gradChargesMM.empty() ? mm_.MMcharges.data() : gradChargesMM.data());
  /* The charges of all of the MM atoms, for the reciprocal space. With the rules of the
   * potential and GMX_QMMM_POT_SCHEME=AMBER these carry the shift of the MM1 charges, which
   * calculate_LR_QM_MM() prepares once; it has run before this routine in every step.
   */
  if (potentialRules && !potChargeShift.empty() && potShiftFull.empty()
      && variant == eqmmmPME)
  {
      potShiftFull.resize(mm_.nrMMatoms_full);
      for (int j = 0; j < mm_.nrMMatoms_full; j++)
      {
          potShiftFull[j] = potChargeShiftOf(mm_.indexMM_full[j]) * mm_.scalefactor;
      }
  }
  if (potentialRules && !potShiftFull.empty() && variant == eqmmmPME)
  {
      potFullWork.resize(mm_.nrMMatoms_full);
      for (int j = 0; j < mm_.nrMMatoms_full; j++)
      {
          potFullWork[j] = mm_.MMcharges_full[j] + potShiftFull[j];
      }
  }
  else if (potentialRules)
  {
      potFullWork.clear();
  }
  const real* qMMfull =
          potentialRules
                  ? (potFullWork.empty() ? mm_.MMcharges_full.data() : potFullWork.data())
                  : (gradChargesMMfull.empty() ? mm_.MMcharges_full.data() : gradChargesMMfull.data());
  // The per-pair factor: the exclusions of the gradient, or the MM1 atoms zeroed by a
  //   boundary charge scheme.
  const auto scaleOf = [this, potentialRules](int j, int k) {
      return potentialRules ? mm[0].qmmmScaleFactorPot(j, k) : mm[0].qmmmScaleFactorGrad(j, k);
  };
  real        rcoul = qm_.rcoulomb;
  real        ewaldcoeff_q = qm_.ewaldcoeff_q;
  int         n = qm_.nrQMatoms;
  int         ne = mm_.nrMMatoms;
  int         ne_full = mm_.nrMMatoms_full;
  rvec bond;

  /* all of the contributions to the gradients are calculated in, or immediately converted to,
   * ATOMIC UNITS!
   */

  for (int j=0; j<ne; j++)
  {
      clear_rvec(MMgrad[j]);
  }
  if (variant == eqmmmPME)
  {
    for (int j=0; j<ne_full; j++)
    {
      clear_rvec(MMgrad_full[j]);
    }
  }

  switch (variant)
  {

    case eqmmmSWITCH:
    {
      real r_1 = rcoul;
      real r_d = QMMM_SWITCH;
      real r_c = r_1 + r_d;
      real big_a =   (5 * r_c - 2 * r_1) / (CUB(r_c) * SQR(r_d));
      real big_b = - (4 * r_c - 2 * r_1) / (CUB(r_c) * CUB(r_d));
      for (int j=0; j<n; j++) { // do it for every QM atom
        // add SR potential only from MM atoms in the neighbor list!
        for (int k=0; k<ne; k++) {
          // charge scaled down (or zeroed) by the exclusions of the gradient
          const real qMM = qMMsr[k] * scaleOf(j, k);
          if (qMM == 0.)
          {
              continue;
          }
          pbc_dx_qmmm(qm_.box, qm_.xQM[j], mm_.xMM[k], bond);
          real r = norm(bond);
          rvec dgr;
          if (r < 0.001)
          { // this may occur on the first step of simulation for link atom(s)
              printf("QM/MM PME QM--MM short range exploding for QM=%d, MM=%d. MM charge is %f\n", j+1, k+1, mm_.MMcharges[k]);
              continue;
          }
          if (r < r_1)
          {
              real fscal = - qQM[j] * qMM / CUB(r) * SQR(BOHR2NM);
              svmul(fscal, bond, dgr);
              //printf("SR: QM %1d -- MM %1d:%12.7f%12.7f%12.7f\n", j+1, k+1, dgr[XX], dgr[YY], dgr[ZZ]);
              rvec_inc(partgrad[j], dgr);
              rvec_dec(MMgrad[k], dgr);
              continue;
          }
          if (r < r_c)
          {
              real fscal = - qQM[j] * qMM / r * (1. / SQR(r)
                           - big_a * SQR(r - r_1) - big_b * CUB(r - r_1)) * SQR(BOHR2NM);
              svmul(fscal, bond, dgr);
              rvec_inc(partgrad[j], dgr);
              rvec_dec(MMgrad[k], dgr);
          	continue;
          }
          // else ... beyond cutoff+switch, nothing to do
        } // for k
      } // for j
      break;
    }

    case eqmmmRFIELD:
    {
      real r_c = rcoul;
      for (int j=0; j<n; j++) { // do it for every QM atom
        // add SR potential only from MM atoms in the neighbor list!
        for (int k=0; k<ne; k++) {
          // charge scaled down (or zeroed) by the exclusions of the gradient
          const real qMM = qMMsr[k] * scaleOf(j, k);
          if (qMM == 0.)
          {
              continue;
          }
          pbc_dx_qmmm(qm_.box, qm_.xQM[j], mm_.xMM[k], bond);
          real r = norm(bond);
          rvec dgr;
          if (r < 0.001)
          { // this may occur on the first step of simulation for link atom(s)
              printf("QM/MM PME QM--MM short range exploding for QM=%d, MM=%d. MM charge is %f\n", j+1, k+1, mm_.MMcharges[k]);
              continue;
          }
          if (r < r_c)
          {
              real fscal = - qQM[j] * qMM / r * (1. / SQR(r) - r / CUB(r_c)) * SQR(BOHR2NM);
              svmul(fscal, bond, dgr);
              rvec_inc(partgrad[j], dgr);
              rvec_dec(MMgrad[k], dgr);
              continue;
          }
          // else ... beyond cutoff, nothing to do
        } // for k
      } // for j
      break;
    }

    case eqmmmSHIFT:
    {
      real r_c = rcoul;
      real big_c = 3. / SQR(r_c);
      for (int j=0; j<n; j++) { // do it for every QM atom
        // add SR potential only from MM atoms in the neighbor list!
        for (int k=0; k<ne; k++) {
          // charge scaled down (or zeroed) by the exclusions of the gradient
          const real qMM = qMMsr[k] * scaleOf(j, k);
          if (qMM == 0.)
          {
              continue;
          }
          pbc_dx_qmmm(qm_.box, qm_.xQM[j], mm_.xMM[k], bond);
          real r = norm(bond);
          rvec dgr;
          if (r < 0.001)
          { // this may occur on the first step of simulation for link atom(s)
              printf("QM/MM PME QM--MM short range exploding for QM=%d, MM=%d. MM charge is %f\n", j+1, k+1, mm_.MMcharges[k]);
              continue;
          }
          if (r < r_c)
          {
              real fscal = - qQM[j] * qMM / r * (1. / SQR(r) + 2. * r / CUB(r_c) - big_c) * SQR(BOHR2NM);
              svmul(fscal, bond, dgr);
              rvec_inc(partgrad[j], dgr);
              rvec_dec(MMgrad[k], dgr);
              continue;
          }
        } // for k
      } // for j
      break;
    }

    case eqmmmPME: // QM/MM with PME 
    {
      QMMM_PME& pme_full   = pme[0];
      QMMM_PME& pme_qmonly = pme[1];
    //std::vector<rvec> grad_add(n);
      rvec* grad_add = new rvec [n];

      /* (1) gradient on QM atoms due to MM atoms. */

      // copy the data into PME structures
      for (int j=0; j<n; j++)
      {
          pme_full.x[j][XX] = qm_.xQM[j][XX];
          pme_full.x[j][YY] = qm_.xQM[j][YY];
          pme_full.x[j][ZZ] = qm_.xQM[j][ZZ];
          /* unscaled QM charges.
           * ATTENTION -- the interaction of periodic QM images will be included FULLY,
           * and thus it has to be reduced below, to account for the possibly requested scalefactor.
           */
          pme_full.q[j]     = qQM[j];
      }
      for (int j=0; j<ne_full; j++)
      { 
          pme_full.x[n + j][XX] = mm_.xMM_full[j][XX];
          pme_full.x[n + j][YY] = mm_.xMM_full[j][YY];
          pme_full.x[n + j][ZZ] = mm_.xMM_full[j][ZZ];
          /* the MM charges are already scaled */
          pme_full.q[n + j]     = qMMfull[j];
      }
      // PME -- long-range component
    //static struct timespec time1, time2;
    //clock_gettime(CLOCK_MONOTONIC, &time1);
      // init_nrnb(pme_full.nrnb); // TODO change to something?
      gmx::StepWorkload stepWork;
      stepWork.computeForces = true;
      std::vector<real> emptyVec;
      /* The exact reciprocal-space virial, when this step needs one: the virial of the grid
       * energy of all of the charges here, minus that of the MM charges alone (below), is the
       * virial of the reciprocal-space QM--MM and QM--QM-image energy. It replaces the single
       * sum x (x) F of the reciprocal-space forces, which is not the virial of an Ewald sum.
       */
      const bool wantVirial = virialCorrection && computeVirial;
      matrix recipVirAll, recipVirMM, recipSingleSum;
      clear_mat(recipVirAll);
      clear_mat(recipVirMM);
      clear_mat(recipSingleSum);
      real recipEnergy = 0, recipDvdl = 0;
      stepWork.computeVirial = wantVirial;
      stepWork.computeEnergy = wantVirial;
      gmx_pme_do(pmedata, pme_full.x, pme_full.f, pme_full.q.data(), pme_full.q.data(),
                 nullptr, nullptr, nullptr, nullptr, qm_.box, cr, 0, 0, nrnb, // pme_full.nrnb->get(),
                 wcycle, recipVirAll, recipVirAll, &recipEnergy, &recipEnergy, 0., 0., &recipDvdl, &recipDvdl,
                 stepWork, TRUE, FALSE, n, nullptr); // emptyVec);
      stepWork.computeVirial = false;
      stepWork.computeEnergy = false;
      for (int j=0; j<n; j++)
      {
        for (int m=0; m<DIM; m++)
        {
          grad_add[j][m] = - pme_full.f[j][m] / HARTREE_BOHR2MD; // partgrad is gradient, i.e. the negative of force
        }
      }
      if (wantVirial)
      {
        rvec image;
        for (int j=0; j<n; j++)
        {
          qmmmNearestImage(qm_.box, qm_.xQM[0], qm_.xQM[j], image);
          qmmmAddOuter(recipSingleSum, image, pme_full.f[j]);
        }
        // the grid virial of the MM charges alone, which is not part of the QM/MM energy
        for (int j=0; j<n; j++)
        {
          pme_full.q[j] = 0.;
        }
        gmx::StepWorkload stepWorkMM;
        stepWorkMM.computeVirial = true;
        stepWorkMM.computeEnergy = true;
        gmx_pme_do(pmedata, pme_full.x, pme_full.f, pme_full.q.data(), pme_full.q.data(),
                   nullptr, nullptr, nullptr, nullptr, qm_.box, cr, 0, 0, nrnb,
                   wcycle, recipVirMM, recipVirMM, &recipEnergy, &recipEnergy, 0., 0., &recipDvdl, &recipDvdl,
                   stepWorkMM, TRUE, FALSE, n, nullptr);
        for (int j=0; j<n; j++)
        {
          pme_full.q[j] = qQM[j];
        }
      }
   // printf("================================\n");
   // for (int i=0; i<n; i++)
   // {
   //     printf("GRAD QMMM1 %d: %8.5f %8.5f %8.5f\n", i+1, grad_add[i][XX], grad_add[i][YY], grad_add[i][ZZ]);
   // }
      for (int j=0; j<n; j++) {
        rvec_inc(partgrad[j], grad_add[j]);
      }

      // PME corrections -- exclude QM--QM interaction within the basis PBC cell
      for (int j=0; j<n; j++) {
        clear_rvec(grad_add[j]);
      }
      for (int j=0; j<n; j++) {
        // Exclude the QM[j]--QM[k] interactions:
        for (int k=0; k<j; k++) {
          rvec_sub(qm_.xQM[j], qm_.xQM[k], bond);
          real r = norm(bond);
          rvec dgr;
          // negative of gradient -- we want to subtract it from partgrad
          real fscal = qQM[j] * qQM[k] / SQR(r) *
                        (gmx_erf(ewaldcoeff_q * r) / r
                       - M_2_SQRTPI * ewaldcoeff_q * exp(-SQR(ewaldcoeff_q * r))) * SQR(BOHR2NM);
          svmul(fscal, bond, dgr); // vec(dgr) = fscal * vec(bond)
          rvec_inc(grad_add[j], dgr);
          rvec_dec(grad_add[k], dgr);
        }
      }
      for (int j=0; j<n; j++) {
        rvec_inc(partgrad[j], grad_add[j]);
      }
   // printf("================================\n");
   // for (int i=0; i<n; i++)
   // {
   //     printf("GRAD CORR1 %d: %8.5f %8.5f %8.5f\n", i+1, grad_add[i][XX], grad_add[i][YY], grad_add[i][ZZ]);
   // }

      /* (2) gradient on QM atoms due to the periodic images of QM atoms.
       *     This is contained in the above contribution already, however with unscaled QM charges.
       *     The current contribution (2) is meant to correct for this,
       *       therefore it will be scaled with (scalefactor-1) at the end of the calculation!
       */

      // copy the data into PME structures
      for (int j=0; j<n; j++)
      {
          pme_qmonly.x[j][XX] = qm_.xQM[j][XX];
          pme_qmonly.x[j][YY] = qm_.xQM[j][YY];
          pme_qmonly.x[j][ZZ] = qm_.xQM[j][ZZ];
          /* unscaled QM charges; the resulting gradients will be scaled down at the end of the calculation! */
          pme_qmonly.q[j]     = qQM[j];
      }
      // PME -- long-range component
    //clock_gettime(CLOCK_MONOTONIC, &time1);
      // init_nrnb(pme_qmonly.nrnb); // TODO change to something?
      int oldNumAtoms = pmedata->atc[0].numAtoms(); // need to resize PME arrays to the number of QM atoms
      pmedata->atc[0].setNumAtoms(n);
      gmx_pme_do(pmedata, pme_qmonly.x, pme_qmonly.f, pme_qmonly.q.data(), pme_qmonly.q.data(),
                 nullptr, nullptr, nullptr, nullptr, qm_.box, cr, 0, 0, // pme_full.nrnb->get(),
                 nrnb, wcycle, pme_qmonly.vir, pme_qmonly.vir, nullptr, nullptr, 0., 0., nullptr, nullptr,
                 stepWork, TRUE, FALSE, n, nullptr); // emptyVec);
      pmedata->atc[0].setNumAtoms(oldNumAtoms); // resize back
    //clock_gettime(CLOCK_MONOTONIC, &time2);
    //print_time_difference("PMETIME 4 ", time1, time2);
      for (int j=0; j<n; j++)
      {
        for (int m=0; m<DIM; m++)
        {
          grad_add[j][m] = - pme_qmonly.f[j][m] / HARTREE_BOHR2MD * (mm_.scalefactor - 1.);
        }
        /* This reciprocal-space part used to be overwritten below without ever reaching
         * partgrad, which silently dropped it. It vanishes for scalefactor == 1, so only
         * a run with MMChargeScaleFactor != 1 was affected.
         */
        rvec_inc(partgrad[j], grad_add[j]);
      }

      // PME corrections -- exclude QM--QM interaction within the basis PBC cell
      for (int j=0; j<n; j++) {
        clear_rvec(grad_add[j]);
      }
      for (int j=0; j<n; j++) {
        // Exclude the QM[j]--QM[k] interactions:
        for (int k=0; k<j; k++) {
          rvec_sub(qm_.xQM[j], qm_.xQM[k], bond);
          real r = norm(bond);
          rvec dgr;
          // negative of gradient -- we want to subtract it from partgrad
          real fscal = qQM[j] * qQM[k] / SQR(r) *
                        (gmx_erf(ewaldcoeff_q * r) / r
                       - M_2_SQRTPI * ewaldcoeff_q * exp(-SQR(ewaldcoeff_q * r))) * SQR(BOHR2NM);
          svmul(fscal, bond, dgr); // vec(dgr) = fscal * vec(bond)
          rvec_inc(grad_add[j], dgr);
          rvec_dec(grad_add[k], dgr);
        }
      }
      for (int j=0; j<n; j++) {
        for (int m=0; m<DIM; m++) {
          grad_add[j][m] *= (mm_.scalefactor - 1.);
        }
        rvec_inc(partgrad[j], grad_add[j]);
      }
   // printf("================================\n");
   // for (int i=0; i<n; i++)
   // {
   //     printf("GRAD CORR2 %d: %8.5f %8.5f %8.5f\n", i+1, grad_add[i][XX], grad_add[i][YY], grad_add[i][ZZ]);
   // }
      // TODO: RE-TEST THIS CONTRIBUTION WITH SCALEFACTOR != 1 !!!

      /* (3) gradient on MM atoms due to QM atoms.
       *     ASK GERRIT IF THIS IS REALLY NOT INCLUDED IN GROMACS MM CALCULATIONS!
       *     But I still think it is not included.
       */

      // copy the data into PME structures
      for (int j=0; j<n; j++)
      {
          pme_full.x[j][XX] = qm_.xQM[j][XX];
          pme_full.x[j][YY] = qm_.xQM[j][YY];
          pme_full.x[j][ZZ] = qm_.xQM[j][ZZ];
          pme_full.q[j]     = qQM[j];
      }
      for (int k=0; k<ne_full; k++)
      {
          pme_full.x[n + k][XX] = mm_.xMM_full[k][XX];
          pme_full.x[n + k][YY] = mm_.xMM_full[k][YY];
          pme_full.x[n + k][ZZ] = mm_.xMM_full[k][ZZ];
          pme_full.q[n + k]     = 0.;
      }
      // PME -- long-range component
    //clock_gettime(CLOCK_MONOTONIC, &time1);
      // init_nrnb(pme_full.nrnb); // TODO change to something?
      gmx_pme_do(pmedata, pme_full.x, pme_full.f, pme_full.q.data(), pme_full.q.data(),
                 nullptr, nullptr, nullptr, nullptr, qm_.box, cr, 0, 0, // pme_full.nrnb->get(),
                 nrnb, wcycle, pme_full.vir, pme_full.vir, nullptr, nullptr, 0., 0., nullptr, nullptr,
                 stepWork, TRUE, TRUE, n, nullptr); // emptyVec);
    //clock_gettime(CLOCK_MONOTONIC, &time2);
    //print_time_difference("PMETIME 5 ", time1, time2);
      for (int j=0; j<ne_full; j++)
      {
          MMgrad_full[j][XX] = - qMMfull[j] * pme_full.f[n + j][XX] / HARTREE_BOHR2MD;
          MMgrad_full[j][YY] = - qMMfull[j] * pme_full.f[n + j][YY] / HARTREE_BOHR2MD;
          MMgrad_full[j][ZZ] = - qMMfull[j] * pme_full.f[n + j][ZZ] / HARTREE_BOHR2MD;
      } // svmul(- mm_.MMcharges_full[j] / HARTREE_BOHR2MD, pme->f[n + j], mm_.grad_full[j]);

      if (wantVirial)
      {
        rvec image, force;
        for (int j=0; j<ne_full; j++)
        {
          qmmmNearestImage(qm_.box, qm_.xQM[0], mm_.xMM_full[j], image);
          svmul(qMMfull[j], pme_full.f[n + j], force);
          qmmmAddOuter(recipSingleSum, image, force);
        }
        /* what calculate_QMMM() has to add on top of its single sum:
         *   (exact reciprocal virial) - (the single sum of the reciprocal forces) */
        for (int a=0; a<DIM; a++)
        {
          for (int b=0; b<DIM; b++)
          {
            recipVirialCorrection[a][b] = recipVirAll[a][b] - recipVirMM[a][b]
                                          + 0.5 * recipSingleSum[a][b];
          }
        }
      }
   // printf("================================\n");
   // for (int i=0; i<ne_full; i++)
   // {
   //     if (norm(mm_.grad_full[i]) > 0.001)
   //     printf("GRAD MM    %d: %8.5f %8.5f %8.5f\n", i+1, mm_.grad_full[i][XX], mm_.grad_full[i][YY], mm_.grad_full[i][ZZ]);
   // }

   // /* (4) Surface correction term for both QM and MM. */
   // if (pme->surf_corr_pme)
   // {
   //     // copy the data into PME structures
   //     for (int j=0; j<n; j++)
   //     {
   //         pme->x[j][XX] = qm_.xQM[j][XX];
   //         pme->x[j][YY] = qm_.xQM[j][YY];
   //         pme->x[j][ZZ] = qm_.xQM[j][ZZ];
   //         pme->q[j]     = qm_.QMcharges[j];
   //     }
   //     for (int k=0; k<ne_full; k++)
   //     {
   //         pme->x[n + k][XX] = mm_.xMM_full[k][XX];
   //         pme->x[n + k][YY] = mm_.xMM_full[k][YY];
   //         pme->x[n + k][ZZ] = mm_.xMM_full[k][ZZ];
   //         pme->q[n + k]     = mm_.MMcharges[k];
   //     }
   //     rvec qx, sum_qx, dgr;
   //     // sum_j q_j vec(x_j)
   //     clear_rvec(sum_qx);
   //     for (int j=0; j<n+ne_full; j++)
   //     {
   //       svmul(pme->q[j], pme->x[j], qx);
   //       rvec_inc(sum_qx, qx);
   //     }
   //     // is it OK that the QM charges have been scaled down possibly??? (mm_.scalefactor)
   //   //svmul(NM2BOHR, sum_qx, sum_qx); // TODO -- CONVERSION?
   //     // contribution to the potential
   //     real vol = qm_.box[XX][XX] * qm_.box[YY][YY] * qm_.box[ZZ][ZZ]; //  * CUB(NM2BOHR); // TODO -- CONVERSION?
   //     // partgrad is gradient, i.e. negative of force
   //     for (int j=0; j<qm_.nrQMatoms; j++)
   //     {
   //       svmul(4. * M_PI / 3. / vol / pme->epsilon_r * qm_.QMcharges[j], sum_qx, dgr);
   //       rvec_inc(partgrad[j], dgr);
   //     }
   //     // do this correction for MM atoms as well
   //     for (int j=0; j<mm_.nrMMatoms_full; j++) 
   //     {
   //       svmul(4. * M_PI / 3. / vol / pme->epsilon_r * mm_.MMcharges_full[j], sum_qx, dgr);
   //       rvec_inc(mm_.grad_full[j], dgr);
   //     }
   // }

      /* (5) QM--MM short-range gradient on both QM and MM atoms
       *       - only consider MM atoms on the neighbor list!
       */
      for (int j=0; j<n; j++) {
        clear_rvec(grad_add[j]);
        for (int k=0; k<ne; k++) { // do it for every QM atom
          pbc_dx_qmmm(qm_.box, qm_.xQM[j], mm_.xMM[k], bond);
          real r = norm(bond);
          rvec dgr;
          if (r < 0.001)
          { // this may occur on the first step of simulation for link atom(s)
              printf("QM/MM PME QM--MM short range exploding for QM=%d, MM=%d. MM charge is %f\n", j+1, k+1, mm_.MMcharges[k]);
              continue;
          }
          const real s = scaleOf(j, k);
          if (r < rcoul)
          {
              real fscal = s * qQM[j] * qMMsr[k] / SQR(r) *
                           (- gmx_erfc(ewaldcoeff_q * r) / r
                            - M_2_SQRTPI * ewaldcoeff_q * exp(-SQR(ewaldcoeff_q * r))) * SQR(BOHR2NM);
              svmul(fscal, bond, dgr);
              rvec_inc(grad_add[j], dgr);
              rvec_dec(MMgrad[k], dgr);
          }
          if (s != 1.)
          {
              /* Counterpart of the reciprocal-space correction applied to the potential
               * in calculate_SR_QM_MM(): remove the fraction (1-s) of the pair term
               * erf(beta*r)/r, which the grid calculation has included in full.
               * Not restricted to r < rcoul, because the reciprocal-space contribution
               * is not either.
               */
              real fscal = (1. - s) * qQM[j] * qMMsr[k] / SQR(r) *
                           (gmx_erf(ewaldcoeff_q * r) / r
                            - M_2_SQRTPI * ewaldcoeff_q * exp(-SQR(ewaldcoeff_q * r))) * SQR(BOHR2NM);
              svmul(fscal, bond, dgr);
              rvec_inc(grad_add[j], dgr);
              rvec_dec(MMgrad[k], dgr);
          }
        }
      }

   // printf("================================\n");
   // for (int i=0; i<n; i++)
   // {
   //     printf("GRAD QMMM5 %d: %8.5f %8.5f %8.5f\n", i+1, grad_add[i][XX], grad_add[i][YY], grad_add[i][ZZ]);
   // }
      for (int j=0; j<n; j++) {
        rvec_inc(partgrad[j], grad_add[j]);
      }

   // printf("================================\n");
   // for (int i=0; i<n; i++)
   // {
   //     if (norm(mm_.grad[i]) > 0.001)
   //     printf("GRAD MM  5 %d: %8.5f %8.5f %8.5f\n", i+1, mm_.grad[i][XX], mm_.grad[i][YY], mm_.grad[i][ZZ]);
   // }

      delete[] grad_add;
      // end of PME
      break;
    }

    default: // it should never get this far
      ;
  } // switch (cutoff_qmmm)

  /* The fictitious point charges of the boundary charge scheme belong to the external
   * potential only, so they contribute to this routine with the rules of the potential.
   */
  if (potentialRules && variant != eqmmmVACUO)
  {
      add_boundary_scheme_gradient(variant, qQM, partgrad, MMgrad);
  }
}

