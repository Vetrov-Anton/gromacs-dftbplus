/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 1991-2000, University of Groningen, The Netherlands.
 * Copyright (c) 2001-2004, The GROMACS development team.
 * Copyright (c) 2013,2014,2015,2016,2017 by the GROMACS development team.
 * Copyright (c) 2018,2019,2020, by the GROMACS development team, led by
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
#include "gmxpre.h"

#include "qmmm.h"

#include "config.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <algorithm>
#include <array>
#include <map>
#include <string>
#include <vector>

#include "gromacs/domdec/domdec_struct.h"
#include "gromacs/ewald/pme.h"
#include "gromacs/ewald/ewald_utils.h"
#include "gromacs/fileio/confio.h"
#include "gromacs/gmxlib/network.h"
#include "gromacs/gmxlib/nrnb.h"
#include "gromacs/math/functions.h"
#include "gromacs/math/units.h"
#include "gromacs/math/vec.h"
#include "gromacs/mdlib/force.h"
#include "gromacs/mdlib/qm_dftbplus.h"
#include "gromacs/mdlib/qm_gamess.h"
#include "gromacs/mdlib/qm_gaussian.h"
#include "gromacs/mdlib/qm_mopac.h"
#include "gromacs/mdlib/qm_orca.h"
#include "gromacs/mdtypes/commrec.h"
#include "gromacs/mdtypes/forceoutput.h"
#include "gromacs/mdtypes/forcerec.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/mdtypes/mdatom.h"
#include "gromacs/mdtypes/nblist.h"
#include "gromacs/nbnxm/grid.h"
#include "gromacs/nbnxm/gridset.h"
#include "gromacs/nbnxm/nbnxm.h"
#include "gromacs/nbnxm/pairlist.h"
#include "gromacs/nbnxm/pairlistset.h"
#include "gromacs/nbnxm/pairlistsets.h"
#include "gromacs/nbnxm/pairsearch.h"
#include "gromacs/pbcutil/ishift.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/topology/ifunc.h"
#include "gromacs/topology/mtop_lookup.h"
#include "gromacs/utility/cstringutil.h"
#include "gromacs/utility/stringutil.h"
#include "gromacs/topology/mtop_util.h"
#include "gromacs/topology/topology.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/smalloc.h"

// When not built in a configuration with QMMM support, much of this
// code is unreachable by design. Tell clang not to warn about it.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunreachable-code"
#pragma GCC diagnostic ignored "-Wmissing-noreturn"

void put_cluster_in_MMlist_verlet(int                            ck, // cluster number
                                  int                            na_ck, // # of atoms in cluster
                                  int                            nrQMatoms,
                                  const int*                     indexQM,
                                        gmx::ArrayRef<const int> atomIndices,
			                      int*                           shiftMMatom,
                                  // ^ also has a role of "bool* isMMatom"
				                  t_pbc*                         pbc,
				                  const rvec*                    x);

/*
std::unique_ptr<QMMM_rec>
void init_QMMM_rec(const t_commrec  *cr,
              const gmx_mtop_t *mtop,
              const t_inputrec *ir,
              const t_forcerec *fr,
              const gmx_wallcycle_t gmx_unused wcycle)
{
    return std::make_unique<QMMM_rec>(cr, mtop, ir, fr, wcycle);
}
*/

static real call_QMroutine(const t_commrec*  cr,
                           QMMM_rec*         qr,
                           QMMM_QMrec*       qm,
                           QMMM_MMrec*       mm,
                           rvec              f[],
                           rvec              fshift[],
                           t_nrnb*           nrnb,
                           gmx_wallcycle_t   wcycle)
{
    // Makes a call to the requested QM routine (qm->QMmethod).
    // Note that f is actually the gradient, i.e. -f

    if (GMX_QMMM_MOPAC)
    {
        return call_mopac(*qm, *mm, f, fshift);
    }
    else if (GMX_QMMM_GAMESS)
    {
        return call_gamess(*qm, *mm, f, fshift);
    }
    else if (GMX_QMMM_GAUSSIAN)
    {
        return qm->gaussian.call_gaussian(*qm, *mm, f, fshift);
    }
    else if (GMX_QMMM_ORCA)
    {
        return call_orca(*qm, *mm, f, fshift);
    }
    else if (GMX_QMMM_DFTBPLUS)
    {
        return call_dftbplus(qr, cr, qm, *mm, f, fshift, nrnb, wcycle);
    }
    else
    {
        gmx_fatal(FARGS, "Unknown QM software -- should never land here :-/");
    }
}

// Update QM and MM coordinates in the QM/MM data structures.
// New version of the function:
// Update the coordinates of the MM atoms on the short-range neighborlist!
// The NBlist needs to have been created previously by either group or Verlet scheme.
void QMMM_rec::update_QMMM_coord(const t_commrec*  cr,
                                 rvec*             shift_vec,
                                 const rvec        x[],
                                 const t_mdatoms*  md,
                                 const matrix      box)
{
    // Shifts the QM and MM atoms into the central box and
    //   stores the shifted coordinates in the coordinate arrays of QMMMrec.
    // These coordinates are passed on the QM subroutines.
    //
    // Only MM atoms up to the distance fr->rcoulomb from the respective
    //   nearest QM atoms are considered;
    // in case fr->rcoulomb == 0. is detected,
    //   all of the MM atoms are considered.

    QMMM_QMrec& qm_ = qm[0];
    QMMM_MMrec& mm_ = mm[0];
    real rcut = qm_.rcoulomb > 0.1 ? qm_.rcoulomb : 999999.; // infinity
    std::vector<bool> isCurrentMMatom;
    isCurrentMMatom.resize(mm_.nrMMatoms_nbl);

    // shift the QM atoms into the central box
    for (int i = 0; i < qm_.nrQMatoms; i++)
    {
        rvec_sub(x[qm_.indexQM[i]], shift_vec[qm_.shiftQM[i]], qm_.xQM[i]);
    }

    // copy box size
    copy_mat(box, qm_.box);

    // initialize PBC for MM coordinate manipulation
    t_pbc pbc;
    ivec null_ivec;
    clear_ivec(null_ivec);
    set_pbc_dd(&pbc, pbcType, DOMAINDECOMP(cr) ? cr->dd->numCells : null_ivec, false, box);

 // for (int s = 0; s < pbc.ntric_vec; s++)
 // {
 //     printf("SHIFT[%2d] = %d %d %d\n", s, pbc.tric_shift[s][0], pbc.tric_shift[s][1], pbc.tric_shift[s][2]);
 //  // printf("SHIFT[%2d] = %8.5f %8.5f %8.5f\n", s, pbc.tric_vec[s][0], pbc.tric_vec[s][1], pbc.tric_vec[s][2]);
 // }

    // DECIDE IF WE WANT TO APPLY A CUTOFF ON THE ATOMS FROM THE SR NEIGHBORLIST!

    // DO WE NEED TO RE-ALLOCATE THE ARRAYS TO BE FILLED?
    //   YES!
    //
    // FIRST, IDENTIFY THE MM ATOMS UP TO CUTOFF AT THIS STEP AND COUNT THEM:

    // Among the atoms found as candidates for being MM atoms in neighborsearching,
    // find those that are within electrostatics cut-off.
    // For the cutoff, use the value "rcut"
    int nrMMatoms = 0;
    for (int i = 0; i < mm_.nrMMatoms_nbl; i++)
    {
	    isCurrentMMatom[i] = false;
	 // printf("DEBUG_MM TEST %5d %5d %2d", i, mm_.indexMM_nbl[i], mm_.shiftMM_nbl[i]);
	    // loop over all QM atoms here
	    for (int q=0; q<qm_.nrQMatoms; q++)
	    {
            rvec bond;
            pbc_dx_aiuc(&pbc, x[qm_.indexQM[q]], x[mm_.indexMM_nbl[i]], bond);
	     // printf(" %8.5f\n", norm(bond));
	        if (norm(bond) < rcut)
            {
	            isCurrentMMatom[i] = true;
	            nrMMatoms++;
	         // printf("DEBUG_MM %5d %5d %2d %6.4f\n", i, mm_.indexMM_nbl[i], mm_.shiftMM_nbl[i], distance);
                break;
            }
	    }
	 // printf("\n");
    }
 // printf("Number of actual    MM atoms in the current MD step               : %d\n", nrMMatoms);

    // ALLOCATION
    mm_.nrMMatoms = nrMMatoms;
    mm_.indexMM.resize(nrMMatoms);
    mm_.MMcharges.resize(nrMMatoms);
    mm_.shiftMM.resize(nrMMatoms);
    mm_.xMM.resizeWithPadding(nrMMatoms);

    int index = 0; // runs over the identified MM atoms
    for (int i = 0; i < mm_.nrMMatoms_nbl; i++)
    {
	    if (isCurrentMMatom[i])
	    {
	        // Add to list!
	        mm_.indexMM[index] = mm_.indexMM_nbl[i];

	        // Also add charge
	        mm_.MMcharges[index] = md->chargeA[mm_.indexMM[index]] * mm_.scalefactor;

            // Having obtained the shift at NS time (update_qmmmrec),
            //   merely copy it here to shiftMM[]
	        mm_.shiftMM[index] = mm_.shiftMM_nbl[i];

	        // one MM atom found => increment index */
	        index++;
	    }
    }

    // The short-range MM list has just been rebuilt, so the scaling factors
    //   of the topologically excluded QM--MM pairs have to be re-mapped onto it.
    update_QMMM_exclusion_scaling(md->nr);

    // also shift the MM atoms into the central box

 //   for (int a=0; a<45; a++)
 //     printf("SHIFT %2d: %7.3f %7.3f %7.3f\n", a,
 //     fr->shift_vec[a][XX], fr->shift_vec[a][YY], fr->shift_vec[a][ZZ]);

    for (int ind = 0; ind < mm_.nrMMatoms; ind++)
    {
        rvec_sub(x[mm_.indexMM[ind]], shift_vec[mm_.shiftMM[ind]], mm_.xMM[ind]);
 //     printf("COORD MM %4d %2d\n", mm_.indexMM[ind], mm_.shiftMM[ind]);
    }

    // For DFTB, also update the coordinates of *all* of the MM atoms,
    //   not only those on the short-range neighborlist.
    // Do not shift the MM atoms into the central box!
    //   It might break the calculation of the surface correction in the Ewald sum.
    if (GMX_QMMM_DFTBPLUS)
    {
        for (int i = 0; i < mm_.nrMMatoms_full; i++)
        {
            copy_rvec(x[mm_.indexMM_full[i]], mm_.xMM_full[i]);
        }
    }
} // update_QMMM_coord

void QMMM_QMrec::init_QMrec(int               grpnr,
                            int               nr,
                            const int*        atomarray,
                            const gmx_mtop_t* mtop,
                            const t_inputrec* ir)
{
    nrQMatoms = nr;
    snew(xQM, nr);
    snew(indexQM, nr);
    snew(shiftQM, nr);
    for (int i = 0; i < nr; i++)
    {
        indexQM[i] = atomarray[i];
    }

    snew(atomicnumberQM, nr);
    int molb = 0;
    for (int i = 0; i < nrQMatoms; i++)
    {
        const t_atom &atom = mtopGetAtomParameters(mtop, indexQM[i], &molb);
        nelectrons        += mtop->atomtypes.atomnumber[atom.type];
        atomicnumberQM[i]  = mtop->atomtypes.atomnumber[atom.type];
    }

    QMcharge      = ir->opts.QMcharge[grpnr];
    multiplicity  = ir->opts.QMmult[grpnr];
    nelectrons   -= ir->opts.QMcharge[grpnr];

    QMmethod      = ir->opts.QMmethod[grpnr];
    QMbasis       = ir->opts.QMbasis[grpnr];

    // hack to prevent gaussian from reinitializing all the time
    gaussian.nQMcpus = 0; // number of CPU's to be used by g01, is set
                          // upon initializing gaussian with init_gaussian()

    rcoulomb      = ir->rcoulomb;
    ewaldcoeff_q  = calc_ewaldcoeff_q(ir->rcoulomb, ir->ewald_rtol);
    epsilon_r     = ir->epsilon_r;

    snew(pot_qmmm, nr);
    snew(pot_qmqm, nr);

} // init_QMrec

int QMMM_QMrec::nrQMatoms_get() const
{
    return nrQMatoms;
}

int QMMM_QMrec::qmmm_variant_get() const
{
    return qmmm_variant;
}

double QMMM_QMrec::xQM_get(const int atom, const int coordinate) const
{
    return xQM[atom][coordinate];
}

real QMMM_QMrec::QMcharges_get(const int atom) const
{
    return QMcharges[atom];
}

void QMMM_QMrec::QMcharges_set(const int atom, const real value)
{
    QMcharges[atom] = value;
}

double QMMM_QMrec::pot_qmmm_get(const int atom) const
{
    return pot_qmmm[atom];
}

double QMMM_QMrec::pot_qmqm_get(const int atom) const
{
    return pot_qmqm[atom];
}

void QMMM_QMrec::pot_qmmm_set(const int atom, const double value)
{
    pot_qmmm[atom] = value;
}

void QMMM_QMrec::pot_qmqm_set(const int atom, const double value)
{
    pot_qmqm[atom] = value;
}

int QMMM_QMrec::atomicnumberQM_get(const int atom)const
{
    return atomicnumberQM[atom];
}

int QMMM_QMrec::QMcharge_get()const
{
    return QMcharge;
}

int QMMM_QMrec::multiplicity_get()const
{
    return multiplicity;
}

int QMMM_QMrec::QMmethod_get()const
{
    return QMmethod;
}

int QMMM_QMrec::QMbasis_get()const
{
    return QMbasis;
}

int QMMM_QMrec::nelectrons_get()const
{
    return nelectrons;
}

int QMMM_QMrec::CASelectrons_get()const
{
    return CASelectrons;
}

int QMMM_QMrec::CASorbitals_get()const
{
    return CASorbitals;
}

real QMMM_QMrec::box_xx_get() const
{
    return box[0][0];
}

real QMMM_QMrec::box_yy_get() const
{
    return box[1][1];
}

real QMMM_QMrec::box_zz_get() const
{
    return box[2][2];
}

void QMMM_MMrec::init_MMrec(real scalefactor_in,
                            int  nrMMatoms_full_in,
                            int  natoms,
                            int  nrQMatoms,
                            const int* indexQM,
                            int* found_mm_atoms)
{
    scalefactor    = scalefactor_in;
    nrMMatoms_full = nrMMatoms_full_in;
    indexMM_full.resize(nrMMatoms_full); // ???
    xMM_full.resizeWithPadding(nrMMatoms_full); // ???
    MMcharges_full.resize(nrMMatoms_full); // ???
    shiftMM_full.resize(nrMMatoms_full); // ???

    // fill the indexMM_full array
    *found_mm_atoms = 0;
    for (int i=0; i<natoms; i++)
    {
        bool is_mm_atom = true;
        for (int j=0; j<nrQMatoms; j++)
        {
            if (i == indexQM[j])
            {
                 is_mm_atom = false;
            }
        }
        if (is_mm_atom)
        {
            indexMM_full[*found_mm_atoms] = i;
	        (*found_mm_atoms)++;
        }
    }
}

QMMM_rec::QMMM_rec(const t_commrec*                 cr,
                   const gmx_mtop_t*                mtop,
                   const t_inputrec*                ir,
                   const t_forcerec*                fr)
 //                const gmx_wallcycle_t gmx_unused wcycle)
{
#if GMX_QMMM
    // Put the atom numbers of atoms that belong to the QMMM group
    // into an array that will be copied later to QMMMrec->indexQM[..].
    // Also, it will be used to create an index array QMMMrec->bQMMM[],
    // which contains true/false for QM and MM (the other) atoms.

    if (!GMX_QMMM)
    {
        gmx_incons("Compiled without QMMM");
    }

    // issue a fatal if the user wants to run with more than one node
    if (PAR(cr))
    {
        gmx_fatal(FARGS, "QM/MM may not work in parallel due to neighborsearching issues, \
              use a single processor instead!\n");
    }

    // The array bQMMM[] contains true/false for atoms that are QM/not QM.
    // We first set all elements at false.
    // Afterwards we use qm_arr (= MMrec->indexQM) to change
    // the elements corresponding to the QM atoms at true.

    // We take the possibility into account
    // that a user has defined more than one QM group:
    // HOW SHOULD WE PROCEED IN THAT CASE?
    // IT WOULD BE COOL TO BE ABLE TO DO IT!

    // An ugly work-around in case there is only one group.
    // In this case, the whole system is treated as QM.
    // Otherwise, the second group is always the rest of the total system
    //   and is treated as MM.

    // Small problem if there is only QM... so no MM. */

    pbcType = fr->pbcType;

    int numQmmmGroups = ir->opts.ngQM;

    if (numQmmmGroups > 1) {
        fprintf(stderr, "\nQM/MM cannot calculate more than 1 group of atoms at the moment\nExiting!\n\n");
        exit(-1);
    }

    // There are numQmmmGroups groups of QM atoms.
    // Previously, multiple QM groups typically meant
    // that the user wanted to do ONIOM.
    // However, maybe it should also be possible to define
    // more than one QM subsystem with independent neighbourlists.
    // Gerrit Groenhof said he would have to think about that...
    // (11-11-2003)

    std::vector<int> qmmmAtoms = qmmmAtomIndices(*ir, *mtop);

    qm.resize(numQmmmGroups);

    // Standard QMMM (no ONIOM).
    // All layers are merged together, so there is one QM subsystem and one MM subsystem.
    // Also, we set the charges to zero in mtop
    //   to prevent the innerloops from doubly counting the electrostatic QM--MM interaction.
    // TODO: Consider doing this in grompp instead.

    // store QM atoms in the QMrec and initialise
    qm[0].init_QMrec(0, qmmmAtoms.size(), qmmmAtoms.data(), mtop, ir);

    // print the current layer to allow users to check their input
    fprintf(stderr, "Layer %d\nnr of QM atoms %d\n", 0, qm[0].nrQMatoms);
    fprintf(stderr, "QMlevel: %s/%s\n\n",
            eQMmethod_names[qm[0].QMmethod], eQMbasis_names[qm[0].QMbasis]);

    // MM rec creation
    int nrMMatoms_full_in = (mtop->natoms)-(qm[0].nrQMatoms); // rest of the atoms
    int found_mm_atoms = 0;
    mm.resize(1);
    QMMM_MMrec& mm_ = mm[0];
    mm_.init_MMrec(ir->scalefactor, nrMMatoms_full_in, mtop->natoms, qm[0].nrQMatoms, qm[0].indexQM, &found_mm_atoms); 

    printf ("(mtop->natoms) = %d\n(qr->qm[0]->nrQMatoms) = %d\nmm->nrMMatoms_full = %d\n",
            (mtop->natoms), (qm[0].nrQMatoms), mm_.nrMMatoms_full);
    printf ("(found_mm_atoms) = %d\n", found_mm_atoms);

    // Optional topological exclusions of the QM--MM electrostatics.
    init_QMMM_exclusions(mtop, fr, cr);

    // these variables get updated in the update QMMMrec // ???

    // OLD COMMENT but maybe useful in the future:
    //   With only one layer there is only one initialization needed.
    //   Multilayer is a bit more complicated as it requires
    //   a re-initialization at every step of the simulation.
    //   This is due to the use of COMMON blocks in Fortran QM subroutines.

    if (GMX_QMMM_MOPAC)
    {
        init_mopac(qm[0]);
    }
    else if (GMX_QMMM_GAMESS)
    {
        init_gamess(cr, qm[0], mm_);
    }
    else if (GMX_QMMM_GAUSSIAN)
    {
        qm[0].gaussian.init_gaussian();
    }
    else if (GMX_QMMM_ORCA)
    {
        init_orca(&(qm[0]));
    }
    else if (GMX_QMMM_DFTBPLUS)
    {
        // Look how the QM/MM electrostatics shall be treated.
        // In the future, this could be performed for QM/MM in general,
        //   not only with DFTB+.
        char *env1 = getenv("GMX_QMMM_VARIANT");
        char *env2 = getenv("GMX_QMMM_PME_DIPCOR");
        if (env1 == nullptr)
        {
            qm[0].qmmm_variant = eqmmmVACUO;
		    fprintf(stdout, "No electrostatic QM/MM interaction.\nTo change, set environment variable GMX_QMMM_VARIANT.\n");
        }
        else
        {
            sscanf(env1, "%d", &(qm[0].qmmm_variant));
            switch (qm[0].qmmm_variant) {
		    case eqmmmVACUO: // 0
		                    fprintf(stdout, "No electrostatic QM/MM interaction.\n");
		                    break;
		    case eqmmmPME: // 1
                {
		               if (pbcType != PbcType::Xyz)
                       {
		                   fprintf(stderr, "PME treatment of QM/MM electrostatics only possible with triclinic periodic system!\n");
		                   exit(-1);
		               }
		               fprintf(stdout, "Electrostatic QM/MM interaction calculated with full PME treatment.\n");

                       pme.resize(2);
                       pmedata              = &(fr->pmedata);
                       QMMM_PME& pme_full   = pme[0];
                       QMMM_PME& pme_qmonly = pme[1];

                       // PME data structure for the entire system
                       pme_full.x.resizeWithPadding(qm[0].nrQMatoms + mm_.nrMMatoms_full);
                       pme_full.q.resize(qm[0].nrQMatoms + mm_.nrMMatoms_full);
                       pme_full.f.resizeWithPadding(qm[0].nrQMatoms + mm_.nrMMatoms_full);
                       snew(pme_full.pot, qm[0].nrQMatoms);
                       
                       // PME data structure for the QM-only system
                       pme_qmonly.x.resizeWithPadding(qm[0].nrQMatoms);
                       pme_qmonly.q.resize(qm[0].nrQMatoms);
                       pme_qmonly.f.resizeWithPadding(qm[0].nrQMatoms);
                       snew(pme_qmonly.pot, qm[0].nrQMatoms);
                       
                       if (env2 != nullptr)
                       {
                           pme_full.surf_corr_pme   = true;
                           pme_full.epsilon_r       = qm[0].epsilon_r;
                           pme_qmonly.surf_corr_pme = true;
                           pme_qmonly.epsilon_r     = qm[0].epsilon_r;
					       fprintf(stdout, "Dipole (surface) correction for QM/MM PME applied ");
					       fprintf(stdout, "with a permittivity of %5.1f.\n", pme_qmonly.epsilon_r);
					       fprintf(stdout, "\nCurrently disabled due to solvent molecules broken across box boundary!\nExiting!\n\n");
                           exit(-1);
                       }
                       else
                       {
                           pme_full.surf_corr_pme   = false;
                           pme_qmonly.surf_corr_pme = false;
					       fprintf(stdout, "No dipole (surface) correction for QM/MM PME, i.e. tin-foil boundary conditions.\n");
                       }
		               break;
                }
			case eqmmmSWITCH: // 2
		                 fprintf(stdout, "Electrostatic QM/MM interaction calculated with a switched cut-off.\n");
		                 break;
		    case eqmmmRFIELD: // 3
		                 fprintf(stdout, "Electrostatic QM/MM interaction calculated with a reaction-field cut-off.\n");
		                 break;
		    case eqmmmSHIFT: // 4
		                 fprintf(stdout, "Electrostatic QM/MM interaction calculated with a shifted cut-off.\n");
		                 break;
		    default:
		            fprintf(stderr, "Unrecognized choice for treatment of QM/MM electrostatics.\n");
		            fprintf(stderr, "Set environment variable GMX_QMMM_VARIANT to either 0, 1, 2, 3, or 4.\n");
	                exit(-1);
		    }
        }
        snew(qm[0].QMcharges, qm[0].nrQMatoms);

        init_dftbplus(&(qm[0]), this, ir, cr); //, wcycle);
    }
    else
    {
        gmx_fatal(FARGS, "Unknown QM software -- should never land here :-/");
    }
#else // GMX_QMMM
    gmx_incons("Compiled without QMMM");
    (void) cr;
    (void) mtop;
    (void) ir;
    (void) fr;
#endif
} // init_QMMMrec

std::vector<int> qmmmAtomIndices(const t_inputrec& ir, const gmx_mtop_t& mtop)
{
    const int               numQmmmGroups = ir.opts.ngQM;
    const SimulationGroups& groups        = mtop.groups;
    std::vector<int>        qmmmAtoms;
    for (int i = 0; i < numQmmmGroups; i++)
    {
        for (const AtomProxy atomP : AtomRange(mtop))
        {
            int index = atomP.globalAtomNumber();
            if (getGroupType(groups, SimulationAtomGroupType::QuantumMechanics, index) == i)
            {
                qmmmAtoms.push_back(index);
            }
        }
    }
    return qmmmAtoms;
}

void removeQmmmAtomCharges(gmx_mtop_t* mtop, gmx::ArrayRef<const int> qmmmAtoms)
{
    int molb = 0;
    for (gmx::index i = 0; i < qmmmAtoms.ssize(); i++)
    {
        int indexInMolecule;
        mtopGetMolblockIndex(mtop, qmmmAtoms[i], &molb, nullptr, &indexInMolecule);
        t_atom* atom = &mtop->moltype[mtop->molblock[molb].type].atoms.atom[indexInMolecule];
        atom->q      = 0.0;
        atom->qB     = 0.0;
    }
}

// Set up the treatment of the QM/MM boundary in the QM--MM electrostatics.
//
// The external potential passed to DFTB+ and the QM/MM gradient computed by
// Gromacs follow separate rules:
//
// Potential (GMX_QMMM_POT_SCHEME = none, RC, RCD, CS or AMBER): by default every MM
// atom on the short-range list contributes its full point charge. With a boundary
// charge scheme the charge of every MM1 atom (the MM atom from which a link atom
// is constructed) is removed and replaced by fictitious point charges near the
// MM1-MM2 bonds (Lin and Truhlar, J. Phys. Chem. A 109, 3991 (2005) for RC and
// RCD; Sherwood et al., THEOCHEM 632, 1 (2003) for CS), or spread evenly over the
// other MM atoms of their molecule (AMBER). These act on the QM atoms only; the
// topology, the MM interactions and the gradient do not see them.
//
// Gradient (GMX_QMMM_GRAD_EXCL = 0, 1, 2, 3 or BONDED, default 3): the QM--MM pairs
// up to 1-3 are removed and the 1-4 pairs scaled by GMX_QMMM_FUDGE_QQ (default
// fudgeQQ of the force field), with the bonded distance either counted along the
// bond graph or taken from the bonds, angles and proper dihedrals of the tpr.
// The link atoms are excluded as if they were their QM1 atom or, by default, their
// MM1 atom (GMX_QMMM_GRAD_LA).
//
// The bonded neighbour order cannot be recovered from moltype->excls, which is
// a flat list, so it is recomputed here by a breadth-first search over the
// chemical bonds. Note that the bonds inside the QM region were turned into
// F_CONNBONDS by generate_qmexcl_moltype(), which still satisfies IS_CHEMBOND,
// and that the bonds crossing the QM/MM boundary are left intact, so the
// search does see the complete connectivity of the system. The bonds of the
// link atoms (connections, funct 5) are left out: the link atoms are treated
// by GMX_QMMM_GRAD_LA only.
namespace
{

/*! \brief Whether the per-atom QM/MM report files are written.
 *
 * On by default; GMX_QMMM_REPORTS set to 0, no, off or false switches off the
 * reports of both grompp and mdrun, together with the lines that point to them.
 */
bool qmmmReportsEnabled()
{
    const char* env = std::getenv("GMX_QMMM_REPORTS");
    if (env == nullptr)
    {
        return true;
    }
    for (const char* off : { "0", "no", "off", "false" })
    {
        if (gmx_strcasecmp(env, off) == 0)
        {
            return false;
        }
    }
    return true;
}

//! A link atom: a QM virtual site constructed from one QM and one MM atom
struct QmmmLinkAtom
{
    int la;  //!< global index of the link atom
    int qm1; //!< global index of its QM1 atom
    int mm1; //!< global index of its MM1 atom
};

//! A QM--MM pair of the gradient that is removed or scaled
struct QmmmGradPair
{
    int         mm;    //!< global atom index of the MM atom
    real        scale; //!< 0 (removed) or fudgeQQ
    std::string rule;  //!< why: bond distance or bonded term
};

//! "RESnr NAME" label of a global atom
std::string qmmmGlobalAtomLabel(const gmx_mtop_t* mtop, int globalIndex)
{
    int         molb    = 0;
    int         resnr   = 0;
    const char* name    = nullptr;
    const char* resname = nullptr;
    mtopGetAtomAndResidueName(mtop, globalIndex, &molb, &name, &resnr, &resname, nullptr);
    return gmx::formatString("%s%d %s", resname, resnr, name);
}

const char* potSchemeName(QMMM_rec::PotScheme scheme)
{
    switch (scheme)
    {
        case QMMM_rec::PotScheme::RC: return "RC";
        case QMMM_rec::PotScheme::RCD: return "RCD";
        case QMMM_rec::PotScheme::CS: return "CS";
        case QMMM_rec::PotScheme::Amber: return "AMBER";
        default: return "none";
    }
}

//! Name of the link-atom treatment in the gradient, as set with GMX_QMMM_GRAD_LA
const char* gradLaName(QMMM_rec::GradLa la)
{
    switch (la)
    {
        case QMMM_rec::GradLa::QM1: return "QM1";
        case QMMM_rec::GradLa::Exclude: return "exclude";
        default: return "MM1";
    }
}

//! Everything the exclusion report needs, besides the settings in QMMM_rec
struct QmmmReportData
{
    const std::vector<QmmmLinkAtom>*              linkAtoms;
    const std::vector<std::vector<QmmmGradPair>>* gradPairs; // per QM atom
    const std::vector<std::vector<int>>*          bonds;
    const std::vector<bool>*                      bQM;
    const int*                                    indexQM;
    int                                           nrQMatoms;
};

//! Writes, atom by atom, the treatment of the QM--MM electrostatics
void writeQmmmExclusionReport(const gmx_mtop_t* mtop, const QMMM_rec& qr, const QmmmReportData& data, const char* fileName)
{
    FILE* fp = std::fopen(fileName, "w");
    if (fp == nullptr)
    {
        fprintf(stderr, "WARNING: could not open %s for the QM/MM exclusion report\n", fileName);
        return;
    }
    const int*               indexQM   = data.indexQM;
    const int                nrQMatoms = data.nrQMatoms;
    const std::vector<bool>& bQM       = *data.bQM;
    const auto&              bonds = *data.bonds;
    std::vector<bool>        isLA(bQM.size(), false);
    for (const QmmmLinkAtom& l : *data.linkAtoms)
    {
        isLA[l.la] = true;
    }
    const auto atomText = [mtop, &bQM, &isLA](int a) {
        return gmx::formatString("%7d %-14s %s", a + 1, qmmmGlobalAtomLabel(mtop, a).c_str(),
                                 isLA[a] ? "LA" : (bQM[a] ? "QM" : "MM"));
    };
    const auto chargeOf = [mtop](int a) {
        int molb = 0;
        return mtopGetAtomParameters(mtop, a, &molb).q;
    };
    // Bond distances from one atom, up to 7 bonds, for the labels of the tpr section
    const auto distances = [&bonds](int start) {
        std::vector<int> depth(bonds.size(), -1);
        std::vector<int> frontier{ start };
        depth[start] = 0;
        for (int d = 1; d <= 7 && !frontier.empty(); d++)
        {
            std::vector<int> next;
            for (int a : frontier)
            {
                for (int b : bonds[a])
                {
                    if (depth[b] == -1)
                    {
                        depth[b] = d;
                        next.push_back(b);
                    }
                }
            }
            frontier = std::move(next);
        }
        return depth;
    };
    const auto distanceText = [](int d) {
        return d < 0 ? std::string("> 1-8") : gmx::formatString("1-%d", d + 1);
    };

    fprintf(fp, "; QM/MM electrostatics at the QM/MM boundary, written by gmx mdrun\n");
    fprintf(fp, "; atom numbers are global and 1-based, i.e. the numbering of the input .gro file\n");
    fprintf(fp, "; labels are RESIDUEnumber ATOMNAME from the topology; LA = link atom\n");
    fprintf(fp, "; potential passed to DFTB+: GMX_QMMM_POT_SCHEME = %s\n", potSchemeName(qr.potScheme));
    fprintf(fp, "; QM/MM gradient: GMX_QMMM_GRAD_EXCL = %s, GMX_QMMM_FUDGE_QQ = %g, GMX_QMMM_GRAD_LA = %s\n",
            qr.gradBonded ? "BONDED" : gmx::formatString("%d", qr.gradExcl).c_str(), qr.gradFudgeQQ,
            gradLaName(qr.gradLa));
    fprintf(fp, "; the potential and the gradient are independent: a pair listed for one of them\n");
    fprintf(fp, ";   has the full charge in the other one unless listed there as well\n\n");

    fprintf(fp, "[ qm_atoms ]\n; %d atoms, in the order of the QM group\n", nrQMatoms);
    for (int j = 0; j < nrQMatoms; j++)
    {
        fprintf(fp, "%s\n", atomText(indexQM[j]).c_str());
    }

    fprintf(fp, "\n[ link_atoms ]\n; virtual sites of the QM group constructed from a QM1 and an MM1 atom: %zu\n",
            data.linkAtoms->size());
    fprintf(fp, "; %-26s %-26s %s\n", "link atom", "QM1", "MM1");
    for (const QmmmLinkAtom& l : *data.linkAtoms)
    {
        fprintf(fp, "%s %s %s\n", atomText(l.la).c_str(), atomText(l.qm1).c_str(), atomText(l.mm1).c_str());
    }

    // ---- potential ----
    fprintf(fp, "\n[ potential_zeroed_mm1 ]\n");
    if (qr.potScheme == QMMM_rec::PotScheme::None)
    {
        fprintf(fp, "; none: without a boundary charge scheme every MM charge enters the potential in full\n");
    }
    else
    {
        fprintf(fp, "; MM1 atoms whose charge is removed from the potential on every QM atom\n");
        fprintf(fp, "; %-26s %10s\n", "MM1 atom", "q_MM1");
        for (const QmmmLinkAtom& l : *data.linkAtoms)
        {
            fprintf(fp, "%s %+10.5f\n", atomText(l.mm1).c_str(), chargeOf(l.mm1));
        }
    }
    fprintf(fp, "\n[ potential_charge_shift ]\n");
    if (qr.potScheme == QMMM_rec::PotScheme::Amber)
    {
        fprintf(fp, "; AMBER: the MM1 charges are spread over the other MM atoms of their molecule,\n");
        fprintf(fp, ";   for the potential only; every atom that receives a share:\n");
        fprintf(fp, "; %-26s %10s %14s\n", "MM atom", "q_MM", "added");
        for (size_t a = 0; a < qr.potChargeShift.size(); a++)
        {
            if (qr.potChargeShift[a] != real(0.0))
            {
                fprintf(fp, "%s %+10.5f %+14.8e\n", atomText(static_cast<int>(a)).c_str(),
                        chargeOf(static_cast<int>(a)), qr.potChargeShift[a]);
            }
        }
    }
    else
    {
        fprintf(fp, "; none\n");
    }
    fprintf(fp, "\n[ potential_point_charges ]\n");
    if (qr.potPoints.empty())
    {
        fprintf(fp, "; none\n");
    }
    else
    {
        fprintf(fp, "; fictitious charges seen by the QM atoms only, at x(MM1) + f * (x(MM2) - x(MM1));\n");
        fprintf(fp, ";   f = 1 is a change of the charge of MM2 itself, for the potential only\n");
        fprintf(fp, "; %-26s %-26s %6s %10s\n", "MM1 atom", "MM2 atom", "f", "charge");
        for (const QMMM_rec::PotPoint& pt : qr.potPoints)
        {
            fprintf(fp, "%s %s %6.3f %+10.5f\n", atomText(pt.a).c_str(), atomText(pt.b).c_str(), pt.f, pt.q);
        }
    }

    // ---- gradient ----
    const auto writeGradSection = [&](const char* name, const char* comment, bool linkAtoms) {
        fprintf(fp, "\n[ %s ]\n; %s\n", name, comment);
        fprintf(fp, "; %-26s %-26s %10s %8s  %s\n", "QM atom", "MM atom", "q_MM", "factor", "rule");
        int count = 0;
        for (int j = 0; j < nrQMatoms; j++)
        {
            if (isLA[indexQM[j]] != linkAtoms)
            {
                continue;
            }
            for (const QmmmGradPair& p : (*data.gradPairs)[j])
            {
                fprintf(fp, "%s %s %+10.5f %8.4f  %s\n", atomText(indexQM[j]).c_str(),
                        atomText(p.mm).c_str(), chargeOf(p.mm), p.scale, p.rule.c_str());
                count++;
            }
        }
        fprintf(fp, "; %d pairs\n", count);
    };
    writeGradSection("gradient_qm_mm",
                     "QM--MM pairs removed (factor 0) or scaled in the QM/MM gradient; all other pairs "
                     "have factor 1",
                     false);
    writeGradSection("gradient_link_atoms",
                     qr.gradLaExcluded()
                             ? "link atoms: no charge in the gradient, spread over their molecule"
                             : (qr.gradLaAsMM1()
                                        ? "link atoms, excluded as if they were their MM1 atom"
                                        : "link atoms, excluded as if they were their QM1 atom"),
                     true);

    // What the force field keeps at the boundary, as stored in the tpr. The terms that
    //   grompp removed are no longer in the tpr and are listed in the report of grompp.
    std::vector<std::array<int, 2>> ljExcl, lj14;
    int                             offset = 0;
    for (const gmx_molblock_t& molb : mtop->molblock)
    {
        const gmx_moltype_t& molt = mtop->moltype[molb.type];
        for (int mol = 0; mol < molb.nmol; mol++, offset += molt.atoms.nr)
        {
            bool hasQm = false;
            for (int i = 0; i < molt.atoms.nr && !hasQm; i++)
            {
                hasQm = bQM[offset + i];
            }
            if (!hasQm)
            {
                continue;
            }
            for (int i = 0; i < molt.atoms.nr; i++)
            {
                for (int k : molt.excls[i])
                {
                    const int a = offset + i, b = offset + k;
                    if (a < b && bQM[a] != bQM[b])
                    {
                        ljExcl.push_back({ bQM[a] ? a : b, bQM[a] ? b : a });
                    }
                }
            }
            const InteractionList& il = molt.ilist[F_LJ14];
            for (int i = 0; i < il.size(); i += 3)
            {
                const int a = offset + il.iatoms[i + 1], b = offset + il.iatoms[i + 2];
                if (bQM[a] != bQM[b])
                {
                    lj14.push_back({ bQM[a] ? a : b, bQM[a] ? b : a });
                }
            }
        }
    }
    std::sort(ljExcl.begin(), ljExcl.end());
    std::sort(lj14.begin(), lj14.end());
    for (const bool excl : { true, false })
    {
        const auto& list = excl ? ljExcl : lj14;
        fprintf(fp, "\n[ %s ]\n", excl ? "tpr_lj_exclusions_qm_mm" : "tpr_lj14_pairs_qm_mm");
        fprintf(fp, excl ? "; QM--MM pairs excluded from the MM nonbonded interactions (no LJ): %zu\n"
                         : "; QM--MM LJ-14 pairs kept in the force field: %zu\n",
                list.size());
        fprintf(fp, "; for information: this is the force field, not the QM/MM electrostatics above\n");
        fprintf(fp, "; %-26s %-26s %s\n", "QM atom", "MM atom", "bonds");
        int              lastQm = -1;
        std::vector<int> depth;
        for (const auto& pr : list)
        {
            if (pr[0] != lastQm)
            {
                depth  = distances(pr[0]);
                lastQm = pr[0];
            }
            fprintf(fp, "%s %s %s\n", atomText(pr[0]).c_str(), atomText(pr[1]).c_str(),
                    distanceText(depth[pr[1]]).c_str());
        }
    }
    std::fclose(fp);
}

//! The 3-atom angle potentials whose outer atoms are 1-3 neighbours
bool isQmmmAngleType(int ftype)
{
    switch (ftype)
    {
        case F_ANGLES:
        case F_G96ANGLES:
        case F_RESTRANGLES:
        case F_LINEAR_ANGLES:
        case F_CROSS_BOND_BONDS:
        case F_CROSS_BOND_ANGLES:
        case F_UREY_BRADLEY:
        case F_QUARTIC_ANGLES:
        case F_TABANGLES: return true;
        default: return false;
    }
}

//! The proper dihedrals, whose outer atoms are 1-4 neighbours; impropers are not included
bool isQmmmProperDihedralType(int ftype)
{
    switch (ftype)
    {
        case F_PDIHS:
        case F_RBDIHS:
        case F_RESTRDIHS:
        case F_CBTDIHS:
        case F_FOURDIHS:
        case F_TABDIHS: return true;
        default: return false;
    }
}

} // namespace

void QMMM_rec::init_QMMM_exclusions(const gmx_mtop_t* mtop, const t_forcerec* fr, const t_commrec* cr)
{
    QMMM_QMrec& qm_ = qm[0];

    // The bonded interactions at the QM/MM boundary are treated according to the
    //   scheme selected with GMX_QMMM_BONDED_SCHEME in grompp, and the outcome is
    //   already stored in the tpr file. Warn the user who sets the variable here.
    if (getenv("GMX_QMMM_BONDED_SCHEME") != nullptr)
    {
        fprintf(stdout,
                "NOTE: GMX_QMMM_BONDED_SCHEME is evaluated by grompp, not by mdrun.\n"
                "      The treatment of the bonded interactions at the QM/MM boundary is "
                "fixed in the tpr file.\n");
    }

    // ---- settings ----
    const char* env = getenv("GMX_QMMM_POT_SCHEME");
    potScheme       = PotScheme::None;
    if (env != nullptr)
    {
        if (gmx_strcasecmp(env, "none") == 0)
        {
            potScheme = PotScheme::None;
        }
        else if (gmx_strcasecmp(env, "RC") == 0)
        {
            potScheme = PotScheme::RC;
        }
        else if (gmx_strcasecmp(env, "RCD") == 0)
        {
            potScheme = PotScheme::RCD;
        }
        else if (gmx_strcasecmp(env, "CS") == 0)
        {
            potScheme = PotScheme::CS;
        }
        else if (gmx_strcasecmp(env, "AMBER") == 0)
        {
            potScheme = PotScheme::Amber;
        }
        else
        {
            gmx_fatal(FARGS, "GMX_QMMM_POT_SCHEME must be none, RC, RCD, CS or AMBER, but it is '%s'.", env);
        }
    }

    gradExcl   = 3;
    gradBonded = false;
    if ((env = getenv("GMX_QMMM_GRAD_EXCL")) != nullptr)
    {
        char*      end = nullptr;
        const long n   = std::strtol(env, &end, 10);
        if (gmx_strcasecmp(env, "BONDED") == 0)
        {
            gradBonded = true;
        }
        else if (end != env && *end == '\0' && n >= 0 && n <= 3)
        {
            gradExcl = static_cast<int>(n);
        }
        else
        {
            gmx_fatal(FARGS, "GMX_QMMM_GRAD_EXCL must be 0, 1, 2, 3 or BONDED, but it is '%s'.", env);
        }
    }

    // The 1-4 pairs are scaled rather than removed, with the fudge factor
    //   of the force field, unless the user requests a different value.
    gradFudgeQQ = fr->fudgeQQ;
    if ((env = getenv("GMX_QMMM_FUDGE_QQ")) != nullptr)
    {
        char*        end = nullptr;
        const double f   = std::strtod(env, &end);
        if (end == env || *end != '\0')
        {
            gmx_fatal(FARGS, "GMX_QMMM_FUDGE_QQ must be a number, but it is '%s'.", env);
        }
        gradFudgeQQ = static_cast<real>(f);
    }

    gradLa = GradLa::MM1;
    if ((env = getenv("GMX_QMMM_GRAD_LA")) != nullptr)
    {
        if (gmx_strcasecmp(env, "MM1") == 0)
        {
            gradLa = GradLa::MM1;
        }
        else if (gmx_strcasecmp(env, "QM1") == 0)
        {
            gradLa = GradLa::QM1;
        }
        else if (gmx_strcasecmp(env, "exclude") == 0)
        {
            gradLa = GradLa::Exclude;
        }
        else
        {
            gmx_fatal(FARGS, "GMX_QMMM_GRAD_LA must be QM1, MM1 or exclude, but it is '%s'.", env);
        }
    }

    // ---- topology ----
    const int         natoms = mtop->natoms;
    std::vector<bool> bQM(natoms, false);
    std::vector<int>  qmOfAtom(natoms, -1);
    for (int j = 0; j < qm_.nrQMatoms; j++)
    {
        bQM[qm_.indexQM[j]]      = true;
        qmOfAtom[qm_.indexQM[j]] = j;
    }

    // Link atoms: QM virtual sites constructed from one QM and one MM atom.
    std::vector<QmmmLinkAtom> linkAtoms;
    std::vector<bool>         isLA(natoms, false);
    int                       atomOffset = 0;
    for (const gmx_molblock_t& molb : mtop->molblock)
    {
        const gmx_moltype_t& molt = mtop->moltype[molb.type];
        for (int mol = 0; mol < molb.nmol; mol++, atomOffset += molt.atoms.nr)
        {
            for (int ftype = 0; ftype < F_NRE; ftype++)
            {
                if (!IS_VSITE(ftype) || ftype == F_VSITEN)
                {
                    continue;
                }
                const int              nral = NRAL(ftype);
                const InteractionList& il   = molt.ilist[ftype];
                for (int i = 0; i < il.size(); i += 1 + nral)
                {
                    const int site = atomOffset + il.iatoms[i + 1];
                    if (!bQM[site])
                    {
                        continue;
                    }
                    std::vector<int> qmBuild, mmBuild;
                    for (int c = 2; c <= nral; c++)
                    {
                        const int a = atomOffset + il.iatoms[i + c];
                        (bQM[a] ? qmBuild : mmBuild).push_back(a);
                    }
                    if (mmBuild.empty())
                    {
                        continue; // a virtual site inside the QM region
                    }
                    if (qmBuild.size() != 1 || mmBuild.size() != 1)
                    {
                        gmx_fatal(FARGS,
                                  "QM virtual site %d is constructed from MM atoms, but it is not a "
                                  "link atom constructed from one QM and one MM atom.",
                                  site + 1);
                    }
                    linkAtoms.push_back({ site, qmBuild[0], mmBuild[0] });
                    isLA[site] = true;
                }
            }
        }
    }

    // Connectivity of the entire system, in global atom numbering, without the link
    //   atoms; and the bonds, angles and proper dihedrals for GMX_QMMM_GRAD_EXCL=BONDED.
    std::vector<std::vector<int>>   bonds(natoms);
    std::vector<std::array<int, 2>> bondTerms;
    std::vector<std::array<int, 3>> angleTerms;
    std::vector<std::array<int, 4>> dihedralTerms;
    atomOffset = 0;
    for (const gmx_molblock_t& molb : mtop->molblock)
    {
        const gmx_moltype_t& molt = mtop->moltype[molb.type];
        for (int mol = 0; mol < molb.nmol; mol++, atomOffset += molt.atoms.nr)
        {
            for (int ftype = 0; ftype < F_NRE; ftype++)
            {
                const bool isBond     = IS_CHEMBOND(ftype);
                const bool isAngle    = isQmmmAngleType(ftype);
                const bool isDihedral = isQmmmProperDihedralType(ftype);
                if (!isBond && !isAngle && !isDihedral)
                {
                    continue;
                }
                const int              nral = NRAL(ftype);
                const InteractionList& il   = molt.ilist[ftype];
                for (int i = 0; i < il.size(); i += 1 + nral)
                {
                    std::array<int, 4> at = { -1, -1, -1, -1 };
                    bool               la = false;
                    for (int c = 0; c < nral; c++)
                    {
                        at[c] = atomOffset + il.iatoms[i + 1 + c];
                        la    = la || isLA[at[c]];
                    }
                    if (la)
                    {
                        continue;
                    }
                    if (isBond)
                    {
                        bonds[at[0]].push_back(at[1]);
                        bonds[at[1]].push_back(at[0]);
                        bondTerms.push_back({ at[0], at[1] });
                    }
                    else if (isAngle)
                    {
                        angleTerms.push_back({ at[0], at[1], at[2] });
                    }
                    else
                    {
                        dihedralTerms.push_back(at);
                    }
                }
            }
        }
    }

    const auto chargeOf = [mtop](int a) {
        int molb = 0;
        return mtopGetAtomParameters(mtop, a, &molb).q;
    };

    // The terms by atom: every bond of an atom, and the angles and dihedrals in which it
    //   is an outer atom; so that the pairs of an atom are found without scanning the system.
    std::vector<std::vector<int>> bondsOfAtom(natoms), anglesOfAtom(natoms), dihedralsOfAtom(natoms);
    for (size_t t = 0; t < bondTerms.size(); t++)
    {
        bondsOfAtom[bondTerms[t][0]].push_back(static_cast<int>(t));
        bondsOfAtom[bondTerms[t][1]].push_back(static_cast<int>(t));
    }
    for (size_t t = 0; t < angleTerms.size(); t++)
    {
        anglesOfAtom[angleTerms[t][0]].push_back(static_cast<int>(t));
        if (angleTerms[t][2] != angleTerms[t][0])
        {
            anglesOfAtom[angleTerms[t][2]].push_back(static_cast<int>(t));
        }
    }
    for (size_t t = 0; t < dihedralTerms.size(); t++)
    {
        dihedralsOfAtom[dihedralTerms[t][0]].push_back(static_cast<int>(t));
        if (dihedralTerms[t][3] != dihedralTerms[t][0])
        {
            dihedralsOfAtom[dihedralTerms[t][3]].push_back(static_cast<int>(t));
        }
    }

    // ---- potential: boundary charge scheme ----
    mmScalePot.assign(qm_.nrQMatoms, {});
    potPoints.clear();
    potChargeShift.clear();
    if (potScheme == PotScheme::Amber)
    {
        // As in AMBER: the MM1 charges are removed from the potential, and their sum is
        //   added evenly to the other MM atoms of the same molecule.
        // First atom and size of the molecule of every atom
        std::vector<int> molStart(natoms), molSize(natoms);
        int              start = 0;
        for (const gmx_molblock_t& molb : mtop->molblock)
        {
            const int nat = mtop->moltype[molb.type].atoms.nr;
            for (int mol = 0; mol < molb.nmol; mol++, start += nat)
            {
                std::fill(molStart.begin() + start, molStart.begin() + start + nat, start);
                std::fill(molSize.begin() + start, molSize.begin() + start + nat, nat);
            }
        }
        std::vector<bool>   isMM1(natoms, false);
        std::map<int, double> sumOfMolecule; // first atom of the molecule -> sum of its MM1 charges
        for (const QmmmLinkAtom& l : linkAtoms)
        {
            if (isMM1[l.mm1])
            {
                gmx_fatal(FARGS,
                          "GMX_QMMM_POT_SCHEME=AMBER: MM atom %d is the MM1 atom of more than one link "
                          "atom. This is not supported.",
                          l.mm1 + 1);
            }
            isMM1[l.mm1] = true;
            sumOfMolecule[molStart[l.mm1]] += chargeOf(l.mm1);
            for (int j = 0; j < qm_.nrQMatoms; j++)
            {
                mmScalePot[j].emplace_back(l.mm1, real(0.0));
            }
        }
        potChargeShift.assign(natoms, real(0.0));
        for (const auto& mol : sumOfMolecule)
        {
            std::vector<int> receivers;
            for (int a = mol.first; a < mol.first + molSize[mol.first]; a++)
            {
                if (!bQM[a] && !isMM1[a])
                {
                    receivers.push_back(a);
                }
            }
            if (receivers.empty())
            {
                gmx_fatal(FARGS,
                          "GMX_QMMM_POT_SCHEME=AMBER: the molecule of atoms %d-%d has no MM atom to "
                          "spread the MM1 charges to.",
                          mol.first + 1, mol.first + molSize[mol.first]);
            }
            const real shift = static_cast<real>(mol.second / receivers.size());
            for (int a : receivers)
            {
                potChargeShift[a] = shift;
            }
            fprintf(stdout,
                    "QM/MM potential with the boundary charge scheme AMBER: molecule of atoms %d-%d, "
                    "MM1 charges (sum %+.5f) removed and spread over its %zu other MM atoms, %+.5e "
                    "each.\n",
                    mol.first + 1, mol.first + molSize[mol.first], mol.second, receivers.size(), shift);
        }
    }
    else if (potScheme != PotScheme::None)
    {
        std::vector<int> mm1Count(natoms, 0);
        for (const QmmmLinkAtom& l : linkAtoms)
        {
            if (++mm1Count[l.mm1] > 1)
            {
                gmx_fatal(FARGS,
                          "GMX_QMMM_POT_SCHEME=%s: MM atom %d is the MM1 atom of more than one link "
                          "atom. This is not supported.",
                          potSchemeName(potScheme), l.mm1 + 1);
            }
        }
        for (const QmmmLinkAtom& l : linkAtoms)
        {
            std::vector<int> mm2;
            for (int b : bonds[l.mm1])
            {
                if (!bQM[b])
                {
                    mm2.push_back(b);
                }
            }
            if (mm2.empty())
            {
                gmx_fatal(FARGS,
                          "GMX_QMMM_POT_SCHEME=%s: the MM1 atom %d of link atom %d has no MM2 atom to "
                          "redistribute its charge to.",
                          potSchemeName(potScheme), l.mm1 + 1, l.la + 1);
            }
            for (int b : mm2)
            {
                if (mm1Count[b] > 0)
                {
                    gmx_fatal(FARGS,
                              "GMX_QMMM_POT_SCHEME=%s: atom %d is an MM2 atom of MM1 atom %d and an MM1 "
                              "atom itself. This is not supported.",
                              potSchemeName(potScheme), b + 1, l.mm1 + 1);
                }
                for (int c : bonds[b])
                {
                    if (bQM[c])
                    {
                        gmx_fatal(FARGS,
                                  "GMX_QMMM_POT_SCHEME=%s: the MM2 atom %d of MM1 atom %d is bonded to "
                                  "the QM atom %d. This is not supported.",
                                  potSchemeName(potScheme), b + 1, l.mm1 + 1, c + 1);
                    }
                }
            }

            // the charge of MM1 is removed from the potential on every QM atom
            for (int j = 0; j < qm_.nrQMatoms; j++)
            {
                mmScalePot[j].emplace_back(l.mm1, real(0.0));
            }
            const real q0 = chargeOf(l.mm1) / mm2.size();
            for (int b : mm2)
            {
                switch (potScheme)
                {
                    case PotScheme::RC: potPoints.push_back({ l.mm1, b, real(0.5), q0 }); break;
                    case PotScheme::RCD:
                        potPoints.push_back({ l.mm1, b, real(0.5), real(2.0) * q0 });
                        potPoints.push_back({ l.mm1, b, real(1.0), -q0 });
                        break;
                    case PotScheme::CS:
                        potPoints.push_back({ l.mm1, b, real(1.0), q0 });
                        potPoints.push_back({ l.mm1, b, real(0.94), q0 });
                        potPoints.push_back({ l.mm1, b, real(1.06), -q0 });
                        break;
                    default: break;
                }
            }
        }
        fprintf(stdout,
                "QM/MM potential with the boundary charge scheme %s: %zu MM1 charges removed, "
                "%zu fictitious point charges added.\n",
                potSchemeName(potScheme), linkAtoms.size(), potPoints.size());
    }
    else
    {
        fprintf(stdout,
                "QM/MM potential without a boundary charge scheme -- every MM atom within the "
                "cut-off polarizes the QM density.\nTo change, set environment variable "
                "GMX_QMMM_POT_SCHEME to RC, RCD, CS or AMBER.\n");
    }

    // ---- gradient ----
    // Per QM atom: MM atom -> (factor, rule). A removal wins over a scaling.
    std::vector<std::map<int, std::pair<real, std::string>>> grad(qm_.nrQMatoms);
    const auto addGrad = [&grad](int j, int mm, real scale, const std::string& rule) {
        auto it = grad[j].find(mm);
        if (it == grad[j].end())
        {
            grad[j][mm] = { scale, rule };
        }
        else if (scale == real(0.0) && it->second.first != real(0.0))
        {
            it->second = { scale, rule };
        }
    };
    const real fudge = gradFudgeQQ;

    // Along the bond graph, from atom 'start' as seen by QM atom j:
    //   distance d <= min(N,2) removed, d == 3 scaled. The start itself (d = 0) is
    //   only an MM atom for a link atom treated as MM1, and is removed then.
    std::vector<int> depth(natoms, -1); // reused; only the visited entries are reset
    std::vector<int> visited;
    const auto addByDistance = [&](int j, int start) {
        for (int v : visited)
        {
            depth[v] = -1;
        }
        visited.assign(1, start);
        std::vector<int> frontier{ start };
        depth[start] = 0;
        if (!bQM[start])
        {
            addGrad(j, start, real(0.0), "LA as MM1: MM1 itself");
        }
        for (int d = 1; d <= gradExcl; d++)
        {
            std::vector<int> next;
            for (int a : frontier)
            {
                for (int b : bonds[a])
                {
                    if (depth[b] != -1)
                    {
                        continue;
                    }
                    depth[b] = d;
                    next.push_back(b);
                    visited.push_back(b);
                    if (bQM[b])
                    {
                        continue;
                    }
                    addGrad(j, b, d <= 2 ? real(0.0) : fudge, gmx::formatString("1-%d", d + 1));
                }
            }
            frontier = std::move(next);
        }
    };

    // From the bonded terms, for a QM atom: the patterns of the QM/MM boundary
    const auto addQmBonded = [&](int j, int a) {
        for (int ti : bondsOfAtom[a])
        {
            const auto& t = bondTerms[ti];
            if ((t[0] == a && !bQM[t[1]]) || (t[1] == a && !bQM[t[0]]))
            {
                addGrad(j, t[0] == a ? t[1] : t[0], real(0.0), "bond QM-MM");
            }
        }
        for (int ti : anglesOfAtom[a])
        {
            const auto& t = angleTerms[ti];
            for (int dir = 0; dir < 2; dir++)
            {
                const int x = dir ? t[2] : t[0], m = t[1], y = dir ? t[0] : t[2];
                if (x != a || bQM[y])
                {
                    continue;
                }
                addGrad(j, y, real(0.0), bQM[m] ? "angle QM2-QM1-MM1" : "angle QM1-MM1-MM2");
            }
        }
        for (int ti : dihedralsOfAtom[a])
        {
            const auto& t = dihedralTerms[ti];
            for (int dir = 0; dir < 2; dir++)
            {
                const int x = dir ? t[3] : t[0], m1 = dir ? t[2] : t[1], m2 = dir ? t[1] : t[2],
                          y = dir ? t[0] : t[3];
                if (x != a || bQM[y])
                {
                    continue;
                }
                const char* rule = nullptr;
                if (bQM[m1] && bQM[m2])
                {
                    rule = "dihedral QM3-QM2-QM1-MM1";
                }
                else if (bQM[m1] && !bQM[m2])
                {
                    rule = "dihedral QM2-QM1-MM1-MM2";
                }
                else if (!bQM[m1] && !bQM[m2])
                {
                    rule = "dihedral QM1-MM1-MM2-MM3";
                }
                if (rule != nullptr)
                {
                    addGrad(j, y, fudge, rule);
                }
            }
        }
    };

    // From the bonded terms, for a link atom standing in for its MM1 atom
    const auto addMm1Bonded = [&](int j, int mm1) {
        addGrad(j, mm1, real(0.0), "LA as MM1: MM1 itself");
        for (int ti : bondsOfAtom[mm1])
        {
            const auto& t = bondTerms[ti];
            if ((t[0] == mm1 && !bQM[t[1]]) || (t[1] == mm1 && !bQM[t[0]]))
            {
                addGrad(j, t[0] == mm1 ? t[1] : t[0], real(0.0), "LA as MM1: bond MM1-MM2");
            }
        }
        for (int ti : anglesOfAtom[mm1])
        {
            const auto& t = angleTerms[ti];
            if ((t[0] == mm1 && !bQM[t[2]]) || (t[2] == mm1 && !bQM[t[0]]))
            {
                addGrad(j, t[0] == mm1 ? t[2] : t[0], real(0.0), "LA as MM1: angle MM1-X-MM");
            }
        }
        for (int ti : dihedralsOfAtom[mm1])
        {
            const auto& t = dihedralTerms[ti];
            if ((t[0] == mm1 && !bQM[t[3]]) || (t[3] == mm1 && !bQM[t[0]]))
            {
                addGrad(j, t[0] == mm1 ? t[3] : t[0], fudge, "LA as MM1: dihedral MM1-X-X-MM");
            }
        }
    };

    for (int j = 0; j < qm_.nrQMatoms; j++)
    {
        const int a = qm_.indexQM[j];
        if (isLA[a])
        {
            continue;
        }
        if (gradBonded)
        {
            addQmBonded(j, a);
        }
        else
        {
            addByDistance(j, a);
        }
    }
    for (const QmmmLinkAtom& l : linkAtoms)
    {
        const int j = qmOfAtom[l.la];
        if (gradLa == GradLa::Exclude)
        {
            // The link atom carries no charge in the gradient at all, so there is
            //   nothing to exclude pair by pair; its charge is spread below.
            continue;
        }
        if (gradLa == GradLa::MM1)
        {
            if (gradBonded)
            {
                addMm1Bonded(j, l.mm1);
            }
            else
            {
                addByDistance(j, l.mm1);
            }
        }
        else
        {
            for (const auto& entry : grad[qmOfAtom[l.qm1]])
            {
                addGrad(j, entry.first, entry.second.first, "LA as QM1: " + entry.second.second);
            }
        }
    }

    // GMX_QMMM_GRAD_LA=exclude: the link atoms take no part in the QM/MM electrostatic
    //   gradient and in the gradient of the QM periodic images. Their (Mulliken) charge is
    //   not simply dropped, which would change the charge that the MM subsystem sees, but
    //   spread evenly over the MM atoms of their own molecule, in every step.
    gradLaMolecules.clear();
    gradLaMolOfAtom.clear();
    if (gradLa == GradLa::Exclude)
    {
        std::vector<int> molStart(natoms), molSize(natoms);
        int              start = 0;
        for (const gmx_molblock_t& molb : mtop->molblock)
        {
            const int nat = mtop->moltype[molb.type].atoms.nr;
            for (int mol = 0; mol < molb.nmol; mol++, start += nat)
            {
                std::fill(molStart.begin() + start, molStart.begin() + start + nat, start);
                std::fill(molSize.begin() + start, molSize.begin() + start + nat, nat);
            }
        }
        std::map<int, std::vector<int>> laOfMolecule; // first atom of the molecule -> link atoms
        for (const QmmmLinkAtom& l : linkAtoms)
        {
            laOfMolecule[molStart[l.la]].push_back(qmOfAtom[l.la]);
        }
        gradLaMolOfAtom.assign(natoms, -1);
        for (const auto& mol : laOfMolecule)
        {
            GradLaMolecule entry;
            entry.linkAtomsOfQmList = mol.second;
            for (int a = mol.first; a < mol.first + molSize[mol.first]; a++)
            {
                if (!bQM[a])
                {
                    entry.receivers.push_back(a);
                    gradLaMolOfAtom[a] = static_cast<int>(gradLaMolecules.size());
                }
            }
            if (entry.receivers.empty())
            {
                gmx_fatal(FARGS,
                          "GMX_QMMM_GRAD_LA=exclude: the molecule of atoms %d-%d has %zu link atoms "
                          "but no MM atom to spread their charge to.",
                          mol.first + 1, mol.first + molSize[mol.first], mol.second.size());
            }
            fprintf(stdout,
                    "QM/MM gradient with GMX_QMMM_GRAD_LA=exclude: molecule of atoms %d-%d, the charge "
                    "of its %zu link atoms\n  is removed from the gradient and spread over its %zu MM "
                    "atoms in every step.\n",
                    mol.first + 1, mol.first + molSize[mol.first], entry.linkAtomsOfQmList.size(),
                    entry.receivers.size());
            gradLaMolecules.push_back(std::move(entry));
        }
    }

    mmScaleGrad.assign(qm_.nrQMatoms, {});
    std::vector<std::vector<QmmmGradPair>> gradPairs(qm_.nrQMatoms);
    int                                    nExcluded = 0;
    int                                    nScaled   = 0;
    for (int j = 0; j < qm_.nrQMatoms; j++)
    {
        for (const auto& entry : grad[j])
        {
            if (entry.second.first == real(1.0))
            {
                continue;
            }
            mmScaleGrad[j].emplace_back(entry.first, entry.second.first);
            gradPairs[j].push_back({ entry.first, entry.second.first, entry.second.second });
            (entry.second.first == real(0.0) ? nExcluded : nScaled)++;
        }
    }
    fprintf(stdout,
            "QM/MM gradient: GMX_QMMM_GRAD_EXCL = %s, %d QM--MM pairs removed, %d scaled with "
            "GMX_QMMM_FUDGE_QQ = %g;\n  %zu link atoms, GMX_QMMM_GRAD_LA = %s.\n",
            gradBonded ? "BONDED" : gmx::formatString("%d", gradExcl).c_str(), nExcluded, nScaled,
            gradFudgeQQ, linkAtoms.size(), gradLaName(gradLa));

    // Detailed report, atom by atom, in a separate file.
    if ((cr == nullptr || MASTER(cr)) && qmmmReportsEnabled())
    {
        const char* reportFile = getenv("GMX_QMMM_EXCLUSION_REPORT");
        if (reportFile == nullptr)
        {
            reportFile = "qmmm_exclusion_report.txt";
        }
        const QmmmReportData data = { &linkAtoms, &gradPairs, &bonds, &bQM, qm_.indexQM, qm_.nrQMatoms };
        writeQmmmExclusionReport(mtop, *this, data, reportFile);
        fprintf(stdout, "The QM/MM potential scheme and every QM--MM pair of the gradient that is removed or scaled\n"
                        "  are listed in %s (file name set with GMX_QMMM_EXCLUSION_REPORT, switched off with "
                        "GMX_QMMM_REPORTS=off).\n",
                reportFile);
    }
}

// Map the scaling factors of the potential and of the gradient onto the current
//   short-range MM list, together with the AMBER charges of the potential. Has to be
//   redone whenever that list changes, i.e. in every step.
//   Dense factor arrays are used, so that the inner loops over the MM atoms need no search.
// GMX_QMMM_GRAD_LA=exclude: rebuild the charges that the QM/MM gradient and the gradient
//   of the QM periodic images use. The link atoms get zero, and their charge is added
//   evenly to the MM atoms of their molecule, so that the charge of the whole system is
//   unchanged. Nothing is done with the other settings of GMX_QMMM_GRAD_LA, and the
//   external potential passed to DFTB+ is never affected.
void QMMM_rec::update_gradient_charges(int variant)
{
    if (gradLa != GradLa::Exclude)
    {
        return;
    }
    QMMM_QMrec& qm_ = qm[0];
    QMMM_MMrec& mm_ = mm[0];

    gradChargesQM.assign(qm_.QMcharges, qm_.QMcharges + qm_.nrQMatoms);
    // the shift of every molecule with link atoms, from the current Mulliken charges
    std::vector<real> shiftOfMolecule(gradLaMolecules.size(), real(0.0));
    for (size_t m = 0; m < gradLaMolecules.size(); m++)
    {
        double sum = 0.;
        for (int j : gradLaMolecules[m].linkAtomsOfQmList)
        {
            sum += qm_.QMcharges[j];
            gradChargesQM[j] = real(0.0);
        }
        shiftOfMolecule[m] =
                static_cast<real>(sum / gradLaMolecules[m].receivers.size()) * mm_.scalefactor;
    }

    gradChargesMM.resize(mm_.nrMMatoms);
    for (int k = 0; k < mm_.nrMMatoms; k++)
    {
        const int m     = gradLaMolOfAtom[mm_.indexMM[k]];
        gradChargesMM[k] = mm_.MMcharges[k] + (m < 0 ? real(0.0) : shiftOfMolecule[m]);
    }
    if (variant == eqmmmPME)
    {
        gradChargesMMfull.resize(mm_.nrMMatoms_full);
        for (int k = 0; k < mm_.nrMMatoms_full; k++)
        {
            const int m          = gradLaMolOfAtom[mm_.indexMM_full[k]];
            gradChargesMMfull[k] = mm_.MMcharges_full[k] + (m < 0 ? real(0.0) : shiftOfMolecule[m]);
        }
    }
}

void QMMM_rec::update_QMMM_exclusion_scaling(int natoms)
{
    QMMM_QMrec& qm_ = qm[0];
    QMMM_MMrec& mm_ = mm[0];

    // global -> short-range index; only the entries of the previous list are reset
    if (static_cast<int>(mm_.localIndexOfAtom.size()) != natoms)
    {
        mm_.localIndexOfAtom.assign(natoms, -1);
    }
    else
    {
        for (int g : previousIndexMM)
        {
            mm_.localIndexOfAtom[g] = -1;
        }
    }
    for (int k = 0; k < mm_.nrMMatoms; k++)
    {
        mm_.localIndexOfAtom[mm_.indexMM[k]] = k;
    }
    previousIndexMM.assign(mm_.indexMM.begin(), mm_.indexMM.begin() + mm_.nrMMatoms);

    const auto fill = [&](const std::vector<std::vector<std::pair<int, real>>>& lists,
                          std::vector<real>&                                   scale) {
        bool any = false;
        for (const auto& l : lists)
        {
            any = any || !l.empty();
        }
        if (!any)
        {
            scale.clear();
            return;
        }
        scale.assign(static_cast<size_t>(qm_.nrQMatoms) * mm_.nrMMatoms, real(1.0));
        for (int j = 0; j < qm_.nrQMatoms; j++)
        {
            for (const std::pair<int, real>& exc : lists[j])
            {
                const int k = mm_.localIndexOfAtom[exc.first];
                if (k < 0)
                {
                    // An MM atom within 3 bonds of a QM atom is normally well inside
                    //   the cut-off. If it is not, the pair does not contribute to the
                    //   QM/MM interaction in the first place and there is nothing to scale.
                    continue;
                }
                scale[static_cast<size_t>(j) * mm_.nrMMatoms + k] = exc.second;
            }
        }
    };
    fill(mmScalePot, mm_.qmmmScalePot);
    fill(mmScaleGrad, mm_.qmmmScaleGrad);

    // AMBER: the charges of the potential on the short-range list, ready for the inner loops
    potChargesSR.clear();
    if (!potChargeShift.empty())
    {
        potChargesSR.resize(mm_.nrMMatoms);
        for (int k = 0; k < mm_.nrMMatoms; k++)
        {
            potChargesSR[k] = mm_.MMcharges[k] + potChargeShift[mm_.indexMM[k]] * mm_.scalefactor;
        }
    }
}

// Updates the shift and charges of *all of the* MM atoms in QMMMrec.
//   Only with DFTB.
//   (Not nice, should be done in a more elegant way...)
void QMMM_rec::update_QMMMrec_dftb(const t_commrec*  cr,
                                   rvec*             shift_vec,
                                   const rvec        x[],
                                   const t_mdatoms*  md,
                                   const matrix      box)
{
    // INHERITED NOTE: is NOT yet working if there are no PBC.
    // Also in ns.c, simple NS needs to be fixed!
    //   As of 2019, ns.c does not exist any longer.

    // copy pointers
    QMMM_QMrec& qm_ = qm[0]; // in case of normal QMMM, there is only one group
    QMMM_MMrec& mm_ = mm[0];

    // init_pbc(box); needs to be called first, see pbc.h
    ivec null_ivec;
    clear_ivec(null_ivec);
    t_pbc pbc;
    set_pbc_dd(&pbc, pbcType, DOMAINDECOMP(cr) ? cr->dd->numCells : null_ivec, false, box);

 // printf("There are %d QM atoms, namely:", qm_.nrQMatoms);
 // for (int i=0; i<qm_.nrQMatoms; i++)
 //   printf(" %d", qm_.indexQM[i]);
 // printf("\n");

    // Compute the shift for the MM atoms with respect to QM atom [0].
    // TODO: This looks like a viable first guess, but is that correct?
    // Related to the problem of contributions to virial pressure
    //   in a system treated with particle--mesh Ewald.
    rvec crd;
    rvec_sub(x[qm_.indexQM[0]], shift_vec[qm_.shiftQM[0]], crd);
    for (int i=0; i<mm_.nrMMatoms_full; i++) {
        rvec dx;
        mm_.shiftMM_full[i] = pbc_dx_aiuc(&pbc, crd, x[mm_.indexMM_full[i]], dx);
    }

 // // previous version of the loop
 // for (i=0; i<mm_.nrMMatoms; i++) {
 //     ivec dx;
 //     current_shift = pbc_dx_aiuc(&pbc, x[qm_.indexQM[0]], x[mm_.indexMM[i]], dx);
 //     crd[0] = IS2X(QMMMlist->shift[i]) + IS2X(qm_i_particles[i].shift);
 //     crd[1] = IS2Y(QMMMlist->shift[i]) + IS2Y(qm_i_particles[i].shift);
 //     crd[2] = IS2Z(QMMMlist->shift[i]) + IS2Z(qm_i_particles[i].shift);
 //     is     = static_cast<int>(XYZ2IS(crd[0], crd[1], crd[2]));
 //     mm_.shiftMM[i] = is;
 // }

    for (int i = 0; i < mm_.nrMMatoms_full; i++) // no free energy yet
    {
        mm_.MMcharges_full[i] = md->chargeA[mm_.indexMM_full[i]] * mm_.scalefactor;
    }
} // update_QMMMrec_dftb

// ADD THE NON-QM ATOMS IN THE VERLET CLUSTER ck TO THE LIST OF MM ATOMS
void put_cluster_in_MMlist_verlet(int                            ck, // cluster number
                                  int                            na_ck, // # of atoms in cluster
                                  int                            nrQMatoms,
                                  const int*                     indexQM,
                                  const gmx::ArrayRef<const int> atomIndices,
			                      int*                           shiftMMatom,
                                  // ^ also has a role of "bool* isMMatom"
				                  t_pbc*                         pbc,
				                  const rvec*                    x)
{
 //  * This calculation of shift would be desirable,
 //  * but it does not seem to work properly!
 // ivec crd;
 // crd[XX] = (is_j_cluster ? 1 : -1) * IS2X(shift) + IS2X(qm->shiftQM[qm_atom]);
 // crd[YY] = (is_j_cluster ? 1 : -1) * IS2Y(shift) + IS2Y(qm->shiftQM[qm_atom]);
 // crd[ZZ] = (is_j_cluster ? 1 : -1) * IS2Z(shift) + IS2Z(qm->shiftQM[qm_atom]);
 // int is = IVEC2IS(crd);

    // Loop over the atoms in the cluster ck.
    for (int k=0; k<na_ck; k++)  // NA_CK IS USUALLY 4 (SIMD RELATED)
    {
	    int ck_atom = atomIndices[na_ck * ck + k];
	    if (ck_atom < 0)
	    {
	        // The value of -1 in the Verlet list is for padding purpose only.
	        // It does not correspond to any atom.
	        // Therefore, ignore!
	        continue;
	    }
	    // In the following loop, determine 2 things:
	    // 1: the shift to put the k-th atom (a putative MM atom)
	    //    shortest-distance with respect to the nearest QM atom
	    // 2: whether the k-th atom is a QM atom
	    real dist = 1000.;
	    int sh = -1;
	    bool is_qmatom = false;
	    for (int q=0; q<nrQMatoms; q++)
	    {
	        // 1: the shift -- this calculation looks OK!
	        rvec bond;
	        int sh_t = pbc_dx_aiuc(pbc, x[indexQM[q]], x[ck_atom], bond);
	        if (norm(bond) < dist)
	        {
	            dist = norm(bond);
		        sh = sh_t;
	        }
	        // 2: a QM atom?
	        if (ck_atom == indexQM[q])
	        {
	            is_qmatom = true;
	        }
	    }
	    // If it is not a QM atom, then put it in the list and store the shift.
	    if (!is_qmatom)
	    {
	        // printf("FOUND_MM_ATOM %5d in cluster %4d\n", ck_atom, ck);
	        shiftMMatom[ck_atom] = sh; // true;
	    }
    }
}

// create the SR MM list using the Verlet neighborlist
void QMMM_rec::update_QMMMrec_verlet_ns(const t_commrec*    cr,
                                        nonbonded_verlet_t* nbv,
                                        const rvec          x[],
                                        const t_mdatoms*    md,
                                        const matrix        box)
{
 //  * COMMENTS TO THE FORMER GROUP-SCHEME BASED VERSION OF THIS FUNCTION:
 //  *********************************************************************
 //  * Create/update a number of QMMMrec entries:
 //  * 1) shiftQM -- shifts of the QM atoms
 //  * 2) indexMM -- indices of the MM atoms
 //  * 3) shiftMM -- shifts of the MM atoms
 //  * 4) shifted coordinates of the MM atoms
 //  *       --- NOT THIS ONE, BECAUSE A SEPARATE ROUTINE IS USED!
 //  * (The shifts are used to compute the virial of the QM/MM particles.)
 //  *

 //  * if atom i shall be considered as MM,
 //  *   isMMatom[i] = true
 //  * UPDATE: store the shift in this array,
 //  * and change 'bool' to 'int':
 //  *   shiftMMatom[i] = the value of shift

    std::vector<int> shiftMMatom(md->nr, -1); // ALL ATOMS IN SIMULATION - IS THAT NECESSARY???

    // init PBC
    ivec null_ivec;
    clear_ivec(null_ivec);
    t_pbc pbc;
    set_pbc_dd(&pbc, pbcType, DOMAINDECOMP(cr) ? cr->dd->numCells : null_ivec, false, box);

    // copy pointers
    QMMM_QMrec&                           qm_  = qm[0];
    QMMM_MMrec&                           mm_  = mm[0];
    gmx::ArrayRef<const NbnxnPairlistCpu> nbl = nbv->pairlistSets().pairlistSet(gmx::InteractionLocality::Local).cpuLists();
    int                                   nnbl = nbl.ssize();
    const gmx::ArrayRef<const int>        atomIndices = nbv->pairSearch_->gridSet().atomIndices();

    // QM shift array
    // !!! CHECK THIS !!!
    rvec dx;
    qm_.shiftQM[0] = XYZ2IS(0, 0, 0);
    for (int i = 1; i < qm_.nrQMatoms; i++)
    {
        qm_.shiftQM[i] = pbc_dx_aiuc(&pbc, x[qm_.indexQM[0]], x[qm_.indexQM[i]], dx);
    }
 // for (int i = 0; i < qm->nrQMatoms; i++)
 // {
 //     printf("VERLET QM SHIFT [%d] = %d\n", i, qm->shiftQM[i]);
 // }

    // LOOP OVER THE nnbl NEIGHBORLISTS!
    //   THIS IS NECESSARY WITH MULTITHREADING
    for (int inbl=0; inbl<nnbl; inbl++)
    {
        // loop over CI clusters
        for (unsigned ci=0; ci<nbl[inbl].ci.size(); ci++)
	    {
            // is there a QM atom in this CI cluster?
	        bool qm_atom_in_ci = false;
	        // break the loop if a QM atom has already been found
	        for (int ii=0; ii<nbl[inbl].na_ci && !qm_atom_in_ci; ii++)
	        {
	            // compare to indices of QM atoms
	            for (int iq=0; iq<qm_.nrQMatoms && !qm_atom_in_ci; iq++)
		        {
                    const int iIndex = nbl[inbl].na_ci * nbl[inbl].ci[ci].ci + ii;
                    const int iAtom  = atomIndices[iIndex];
                    //  FORMERLY:
		            // const int iAtom  = nbs->a[nbl[inbl].na_ci * nbl[inbl].ci[ci].ci + ii];
		            if (qm_.indexQM[iq] == iAtom)
		            {
		                qm_atom_in_ci = true;
		            }
		        }
	        }
	        // get the shift of this CI cluster */
	     // shift = nbl[inbl]->ci[ci].shift & NBNXN_CI_SHIFT;

            // loop over the corresponding CJ clusters
	        for (int cj = nbl[inbl].ci[ci].cj_ind_start; cj < nbl[inbl].ci[ci].cj_ind_end; cj++)
	        {
	            // is there a QM atom in this CJ cluster?
	            bool qm_atom_in_cj = false;
	            // break the loop if a QM atom has already been found
	            for (int jj=0; jj<nbl[inbl].na_cj && !qm_atom_in_cj; jj++)
		        {
	                // compare to indices of QM atoms
	                for (int jq=0; jq<qm_.nrQMatoms && !qm_atom_in_cj; jq++)
		            {
                        const int iIndex = nbl[inbl].na_cj * nbl[inbl].cj[cj].cj + jj;
                        const int iAtom  = atomIndices[iIndex];
                        //  FORMERLY:
		                // if (qm->indexQM[jq] == nbs->a[nbl[inbl].na_cj * nbl[inbl].cj[cj].cj + jj])
		                if (qm_.indexQM[jq] == iAtom)
			            {
			                qm_atom_in_cj = true;
		                }
		            }
		        }

                // if there is a QM atom in cluster CI,
		        //   then put the non-QM atoms in cluster CJ into the MM list
	            if (qm_atom_in_ci)
		        {
	                put_cluster_in_MMlist_verlet(nbl[inbl].cj[cj].cj, nbl[inbl].na_cj,
		                                        qm_.nrQMatoms, qm_.indexQM, atomIndices, shiftMMatom.data(), &pbc, x);
	            }

                // if there is a QM atom in cluster CJ,
		        //   then put the non-QM atoms in cluster CI into the MM list
	            if (qm_atom_in_cj)
		        {
	                put_cluster_in_MMlist_verlet(nbl[inbl].ci[ci].ci, nbl[inbl].na_ci,
		                                        qm_.nrQMatoms, qm_.indexQM, atomIndices, shiftMMatom.data(), &pbc, x);
	            }
	        }
	    }
    }

    // count the MM atoms found in the above search
    int nrMMatoms = 0;
    for (int i=0; i<md->nr; i++) {
        // criterium for MM atom found
        if (shiftMMatom[i] != -1)
	    {
	        nrMMatoms++;
	    }
    }

 // printf("Number of potential MM atoms as found in the Verlet neighbor lists: %d\n", nrMMatoms);

    // allocate space and fill the array with atom numbers
    mm_.nrMMatoms_nbl = nrMMatoms;
    mm_.indexMM_nbl.resize(nrMMatoms);
    mm_.shiftMM_nbl.resize(nrMMatoms);
    // index i runs along isMMatom / shiftMMatom,
    //       j runs along the new indexMM_nbl array

    int count=0;
    for (int atom=0; atom<md->nr; atom++) {
        // criterium for MM atom found
        if (shiftMMatom[atom] != -1)
	    {
	        mm_.indexMM_nbl[count] = atom;
	        mm_.shiftMM_nbl[count] = shiftMMatom[atom];
	        count++;
	    }
    }
    // check
    if (count != mm_.nrMMatoms_nbl) {
        printf("ERROR IN MM ATOM SEARCH -- VERLET BASED SCHEME\n");
	    exit(-1);
    }
} // update_QMMMrec_verlet_ns

real QMMM_rec::calculate_QMMM(const t_commrec*      cr,
                              gmx::ForceWithVirial* forceWithVirial,
                              t_nrnb*               nrnb,
                              gmx_wallcycle_t       wcycle)
{
    if (!GMX_QMMM)
    {
        gmx_incons("Compiled without QMMM");
    }

    real QMener = 0.0;
    // A selection for the QM package depending on which is requested
    // (Gaussian, GAMESS-UK, MOPAC or ORCA) needs to be implemented here.
    // Now it works through defines.
    //   ... Not so nice yet

    QMMM_QMrec* qm_ = &(qm[0]);
    QMMM_MMrec* mm_ = &(mm[0]);

    rvec *forces = nullptr,
         *fshift = nullptr;
 
 // gmx::ArrayRef<gmx::RVec> fMM      = forceWithVirial->force_.data();
                  gmx::RVec *fMM      = forceWithVirial->force_.data();
 // gmx::ArrayRef<gmx::RVec> fshiftMM = forceWithShiftForces->shiftForces();

    if (GMX_QMMM_DFTBPLUS)
    {
        snew(forces, (qm_->nrQMatoms + mm_->nrMMatoms + mm_->nrMMatoms_full));
     // snew(fshift, (qm_->nrQMatoms + mm_->nrMMatoms + mm_->nrMMatoms_full));
    }
    else
    {
        snew(forces, (qm_->nrQMatoms + mm_->nrMMatoms));
     // snew(fshift, (qm_.nrQMatoms + mm_.nrMMatoms));
    }

    QMener = call_QMroutine(cr, this, qm_, mm_, forces, fshift, nrnb, wcycle);

    if (GMX_QMMM_DFTBPLUS)
    {
        for (int i = 0; i < qm_->nrQMatoms; i++)
        {
            for (int j = 0; j < DIM; j++)
            {
                fMM[qm_->indexQM[i]][j]        -= forces[i][j];
             // fshiftMM[qm_->shiftQM[i]][j]   += fshift[i][j];
            }
         // printf("F[%5d] = %8.2f %8.2f %8.2f\n", qm_->indexQM[i], forces[i][0], forces[i][1], forces[i][2]);
        }
        for (int i = 0; i < mm_->nrMMatoms; i++)
        {
            for (int j = 0; j < DIM; j++)
            {
                fMM[mm_->indexMM[i]][j]        -= forces[qm_->nrQMatoms+i][j];
             // fshiftMM[mm_->shiftMM[i]][j]   += fshift[qm_->nrQMatoms+i][j];
            }
         // if (i<30) if (norm(forces[qm_->nrQMatoms+i]) > 10.)
         //   printf("F_MM[%5d] = %8.2f %8.2f %8.2f\n", mm_->indexMM[i],
         //     forces[qm_->nrQMatoms+i][0], forces[qm_->nrQMatoms+i][1], forces[qm_->nrQMatoms+i][2]);
        }
        for (int i = 0; i < mm_->nrMMatoms_full; i++)
        {
            for (int j = 0; j < DIM; j++)
            {
                fMM[mm_->indexMM_full[i]][j]        -= forces[qm_->nrQMatoms+mm_->nrMMatoms+i][j];
             // fshiftMM[mm_->shiftMM_full[i]][j]   += fshift[qm_->nrQMatoms+mm_->nrMMatoms+i][j];
            }
         // if (i<100) if (norm(forces[qm_->nrQMatoms+mm_->nrMMatoms+i]) > 10.)
         //   printf("F_MM_F[%5d] = %8.2f %8.2f %8.2f\n", mm_->indexMM_full[i],
         //   forces[qm_->nrQMatoms+mm_->nrMMatoms+i][0], forces[qm_->nrQMatoms+mm_->nrMMatoms+i][1], forces[qm_->nrQMatoms+mm_->nrMMatoms+i][2]);
        }
    }
    else
    {
        for (int i = 0; i < qm_->nrQMatoms; i++)
        {
            for (int j = 0; j < DIM; j++)
            {
                fMM[qm_->indexQM[i]][j]          -= forces[i][j];
             // fshiftMM[qm_->shiftQM[i]][j]     += fshift[i][j];
            }
        }
        for (int i = 0; i < mm_->nrMMatoms; i++)
        {
            for (int j = 0; j < DIM; j++)
            {
                fMM[mm_->indexMM[i]][j]      -= forces[qm_->nrQMatoms+i][j];
             // fshiftMM[mm_->shiftMM[i]][j] += fshift[qm_->nrQMatoms+i][j];
            }
        }
    }

    sfree(forces);
 // sfree(fshift); * but WITHOUT ANY WARRANTY; without even the implied warranty of


    return QMener;
} // calculate_QMMM

#pragma GCC diagnostic pop
