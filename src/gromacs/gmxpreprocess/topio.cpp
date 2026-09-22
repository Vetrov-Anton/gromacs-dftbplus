/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 1991-2000, University of Groningen, The Netherlands.
 * Copyright (c) 2001-2004, The GROMACS development team.
 * Copyright (c) 2013,2014,2015,2016,2017 by the GROMACS development team.
 * Copyright (c) 2018,2019,2020,2021, by the GROMACS development team, led by
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

#include "topio.h"

#include <cassert>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include <sys/types.h>

#include "gromacs/fileio/gmxfio.h"
#include "gromacs/fileio/warninp.h"
#include "gromacs/gmxpreprocess/gmxcpp.h"
#include "gromacs/gmxpreprocess/gpp_atomtype.h"
#include "gromacs/gmxpreprocess/gpp_bond_atomtype.h"
#include "gromacs/gmxpreprocess/gpp_nextnb.h"
#include "gromacs/gmxpreprocess/grompp_impl.h"
#include "gromacs/gmxpreprocess/readir.h"
#include "gromacs/gmxpreprocess/topdirs.h"
#include "gromacs/gmxpreprocess/toppush.h"
#include "gromacs/gmxpreprocess/topshake.h"
#include "gromacs/gmxpreprocess/toputil.h"
#include "gromacs/gmxpreprocess/vsite_parm.h"
#include "gromacs/math/units.h"
#include "gromacs/math/utilities.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/topology/block.h"
#include "gromacs/topology/exclusionblocks.h"
#include "gromacs/topology/ifunc.h"
#include "gromacs/topology/symtab.h"
#include "gromacs/topology/topology.h"
#include "gromacs/utility/cstringutil.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/futil.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/logger.h"
#include "gromacs/utility/pleasecite.h"
#include "gromacs/utility/smalloc.h"
#include "gromacs/utility/stringutil.h"

#define OPENDIR '['  /* starting sign for directive */
#define CLOSEDIR ']' /* ending sign for directive   */

static void gen_pairs(const InteractionsOfType& nbs, InteractionsOfType* pairs, real fudge, int comb)
{
    real scaling;
    int  ntp = nbs.size();
    int  nnn = static_cast<int>(std::sqrt(static_cast<double>(ntp)));
    GMX_ASSERT(nnn * nnn == ntp,
               "Number of pairs of generated non-bonded parameters should be a perfect square");
    int nrfp  = NRFP(F_LJ);
    int nrfpA = interaction_function[F_LJ14].nrfpA;
    int nrfpB = interaction_function[F_LJ14].nrfpB;

    if ((nrfp != nrfpA) || (nrfpA != nrfpB))
    {
        gmx_incons("Number of force parameters in gen_pairs wrong");
    }

    fprintf(stderr, "Generating 1-4 interactions: fudge = %g\n", fudge);
    pairs->interactionTypes.clear();
    int                             i = 0;
    std::array<int, 2>              atomNumbers;
    std::array<real, MAXFORCEPARAM> forceParam = { NOTSET };
    for (const auto& type : nbs.interactionTypes)
    {
        /* Copy type.atoms */
        atomNumbers = { i / nnn, i % nnn };
        /* Copy normal and FEP parameters and multiply by fudge factor */
        gmx::ArrayRef<const real> existingParam = type.forceParam();
        GMX_RELEASE_ASSERT(2 * nrfp <= MAXFORCEPARAM,
                           "Can't have more parameters than half of maximum p  arameter number");
        for (int j = 0; j < nrfp; j++)
        {
            /* If we are using sigma/epsilon values, only the epsilon values
             * should be scaled, but not sigma.
             * The sigma values have even indices 0,2, etc.
             */
            if ((comb == eCOMB_ARITHMETIC || comb == eCOMB_GEOM_SIG_EPS) && (j % 2 == 0))
            {
                scaling = 1.0;
            }
            else
            {
                scaling = fudge;
            }

            forceParam[j]        = scaling * existingParam[j];
            forceParam[nrfp + j] = scaling * existingParam[j];
        }
        pairs->interactionTypes.emplace_back(InteractionOfType(atomNumbers, forceParam));
        i++;
    }
}

double check_mol(const gmx_mtop_t* mtop, warninp* wi)
{
    char   buf[256];
    int    i, ri, pt;
    double q;
    real   m, mB;

    /* Check mass and charge */
    q = 0.0;

    for (const gmx_molblock_t& molb : mtop->molblock)
    {
        const t_atoms* atoms = &mtop->moltype[molb.type].atoms;
        for (i = 0; (i < atoms->nr); i++)
        {
            q += molb.nmol * atoms->atom[i].q;
            m  = atoms->atom[i].m;
            mB = atoms->atom[i].mB;
            pt = atoms->atom[i].ptype;
            /* If the particle is an atom or a nucleus it must have a mass,
             * else, if it is a shell, a vsite or a bondshell it can have mass zero
             */
            if (((m <= 0.0) || (mB <= 0.0)) && ((pt == eptAtom) || (pt == eptNucleus)))
            {
                ri = atoms->atom[i].resind;
                sprintf(buf, "atom %s (Res %s-%d) has mass %g (state A) / %g (state B)\n",
                        *(atoms->atomname[i]), *(atoms->resinfo[ri].name), atoms->resinfo[ri].nr, m, mB);
                warning_error(wi, buf);
            }
            else if (((m != 0) || (mB != 0)) && (pt == eptVSite))
            {
                ri = atoms->atom[i].resind;
                sprintf(buf,
                        "virtual site %s (Res %s-%d) has non-zero mass %g (state A) / %g (state "
                        "B)\n"
                        "     Check your topology.\n",
                        *(atoms->atomname[i]), *(atoms->resinfo[ri].name), atoms->resinfo[ri].nr, m, mB);
                warning_error(wi, buf);
                /* The following statements make LINCS break! */
                /* atoms->atom[i].m=0; */
            }
        }
    }
    return q;
}

/*! \brief Returns the rounded charge of a molecule, when close to integer, otherwise returns the original charge.
 *
 * The results of this routine are only used for checking and for
 * printing warning messages. Thus we can assume that charges of molecules
 * should be integer. If the user wanted non-integer molecular charge,
 * an undesired warning is printed and the user should use grompp -maxwarn 1.
 *
 * \param qMol     The total, unrounded, charge of the molecule
 * \param sumAbsQ  The sum of absolute values of the charges, used for determining the tolerance for the rounding.
 */
static double roundedMoleculeCharge(double qMol, double sumAbsQ)
{
    /* We use a tolerance of 1e-6 for inaccuracies beyond the 6th decimal
     * of the charges for ascii float truncation in the topology files.
     * Although the summation here uses double precision, the charges
     * are read and stored in single precision when real=float. This can
     * lead to rounding errors of half the least significant bit.
     * Note that, unfortunately, we can not assume addition of random
     * rounding errors. It is not entirely unlikely that many charges
     * have a near half-bit rounding error with the same sign.
     */
    double tolAbs = 1e-6;
    double tol    = std::max(tolAbs, 0.5 * GMX_REAL_EPS * sumAbsQ);
    double qRound = std::round(qMol);
    if (std::abs(qMol - qRound) <= tol)
    {
        return qRound;
    }
    else
    {
        return qMol;
    }
}

static void sum_q(const t_atoms* atoms, int numMols, double* qTotA, double* qTotB)
{
    /* sum charge */
    double qmolA    = 0;
    double qmolB    = 0;
    double sumAbsQA = 0;
    double sumAbsQB = 0;
    for (int i = 0; i < atoms->nr; i++)
    {
        qmolA += atoms->atom[i].q;
        qmolB += atoms->atom[i].qB;
        sumAbsQA += std::abs(atoms->atom[i].q);
        sumAbsQB += std::abs(atoms->atom[i].qB);
    }

    *qTotA += numMols * roundedMoleculeCharge(qmolA, sumAbsQA);
    *qTotB += numMols * roundedMoleculeCharge(qmolB, sumAbsQB);
}

static void get_nbparm(char* nb_str, char* comb_str, int* nb, int* comb, warninp* wi)
{
    int  i;
    char warn_buf[STRLEN];

    *nb = -1;
    for (i = 1; (i < eNBF_NR); i++)
    {
        if (gmx_strcasecmp(nb_str, enbf_names[i]) == 0)
        {
            *nb = i;
        }
    }
    if (*nb == -1)
    {
        *nb = strtol(nb_str, nullptr, 10);
    }
    if ((*nb < 1) || (*nb >= eNBF_NR))
    {
        sprintf(warn_buf, "Invalid nonbond function selector '%s' using %s", nb_str, enbf_names[1]);
        warning_error(wi, warn_buf);
        *nb = 1;
    }
    *comb = -1;
    for (i = 1; (i < eCOMB_NR); i++)
    {
        if (gmx_strcasecmp(comb_str, ecomb_names[i]) == 0)
        {
            *comb = i;
        }
    }
    if (*comb == -1)
    {
        *comb = strtol(comb_str, nullptr, 10);
    }
    if ((*comb < 1) || (*comb >= eCOMB_NR))
    {
        sprintf(warn_buf, "Invalid combination rule selector '%s' using %s", comb_str, ecomb_names[1]);
        warning_error(wi, warn_buf);
        *comb = 1;
    }
}

static char** cpp_opts(const char* define, const char* include, warninp* wi)
{
    int         n, len;
    int         ncppopts = 0;
    const char* cppadds[2];
    char**      cppopts   = nullptr;
    const char* option[2] = { "-D", "-I" };
    const char* nopt[2]   = { "define", "include" };
    const char* ptr;
    const char* rptr;
    char*       buf;
    char        warn_buf[STRLEN];

    cppadds[0] = define;
    cppadds[1] = include;
    for (n = 0; (n < 2); n++)
    {
        if (cppadds[n])
        {
            ptr = cppadds[n];
            while (*ptr != '\0')
            {
                while ((*ptr != '\0') && isspace(*ptr))
                {
                    ptr++;
                }
                rptr = ptr;
                while ((*rptr != '\0') && !isspace(*rptr))
                {
                    rptr++;
                }
                len = (rptr - ptr);
                if (len > 2)
                {
                    snew(buf, (len + 1));
                    strncpy(buf, ptr, len);
                    if (strstr(ptr, option[n]) != ptr)
                    {
                        set_warning_line(wi, "mdp file", -1);
                        sprintf(warn_buf, "Malformed %s option %s", nopt[n], buf);
                        warning(wi, warn_buf);
                    }
                    else
                    {
                        srenew(cppopts, ++ncppopts);
                        cppopts[ncppopts - 1] = gmx_strdup(buf);
                    }
                    sfree(buf);
                    ptr = rptr;
                }
            }
        }
    }
    srenew(cppopts, ++ncppopts);
    cppopts[ncppopts - 1] = nullptr;

    return cppopts;
}


static void make_atoms_sys(gmx::ArrayRef<const gmx_molblock_t>      molblock,
                           gmx::ArrayRef<const MoleculeInformation> molinfo,
                           t_atoms*                                 atoms)
{
    atoms->nr   = 0;
    atoms->atom = nullptr;

    for (const gmx_molblock_t& molb : molblock)
    {
        const t_atoms& mol_atoms = molinfo[molb.type].atoms;

        srenew(atoms->atom, atoms->nr + molb.nmol * mol_atoms.nr);

        for (int m = 0; m < molb.nmol; m++)
        {
            for (int a = 0; a < mol_atoms.nr; a++)
            {
                atoms->atom[atoms->nr++] = mol_atoms.atom[a];
            }
        }
    }
}


static char** read_topol(const char*                           infile,
                         const char*                           outfile,
                         const char*                           define,
                         const char*                           include,
                         t_symtab*                             symtab,
                         PreprocessingAtomTypes*               atypes,
                         std::vector<MoleculeInformation>*     molinfo,
                         std::unique_ptr<MoleculeInformation>* intermolecular_interactions,
                         gmx::ArrayRef<InteractionsOfType>     interactions,
                         int*                                  combination_rule,
                         double*                               reppow,
                         t_gromppopts*                         opts,
                         real*                                 fudgeQQ,
                         std::vector<gmx_molblock_t>*          molblock,
                         bool*                                 ffParametrizedWithHBondConstraints,
                         bool                                  bFEP,
                         bool                                  bZero,
                         bool                                  usingFullRangeElectrostatics,
                         warninp*                              wi,
                         const gmx::MDLogger&                  logger)
{
    FILE*                out;
    int                  sl, nb_funct;
    char *               pline = nullptr, **title = nullptr;
    char                 line[STRLEN], errbuf[256], comb_str[256], nb_str[256];
    char                 genpairs[32];
    char *               dirstr, *dummy2;
    int                  nrcopies, nscan, ncombs, ncopy;
    double               fLJ, fQQ, fPOW;
    MoleculeInformation* mi0 = nullptr;
    DirStack*            DS;
    Directive            d, newd;
    t_nbparam **         nbparam, **pair;
    real                 fudgeLJ = -1; /* Multiplication factor to generate 1-4 from LJ */
    bool                 bReadDefaults, bReadMolType, bGenPairs, bWarn_copy_A_B;
    double               qt = 0, qBt = 0; /* total charge */
    int                  dcatt = -1, nmol_couple;
    /* File handling variables */
    int         status;
    bool        done;
    gmx_cpp_t   handle;
    char*       tmp_line = nullptr;
    char        warn_buf[STRLEN];
    const char* floating_point_arithmetic_tip =
            "Total charge should normally be an integer. See\n"
            "http://www.gromacs.org/Documentation/Floating_Point_Arithmetic\n"
            "for discussion on how close it should be to an integer.\n";
    /* We need to open the output file before opening the input file,
     * because cpp_open_file can change the current working directory.
     */
    if (outfile)
    {
        out = gmx_fio_fopen(outfile, "w");
    }
    else
    {
        out = nullptr;
    }

    /* open input file */
    auto cpp_opts_return = cpp_opts(define, include, wi);
    status               = cpp_open_file(infile, &handle, cpp_opts_return);
    if (status != 0)
    {
        gmx_fatal(FARGS, "%s", cpp_error(&handle, status));
    }

    /* some local variables */
    DS_Init(&DS);                   /* directive stack	*/
    d       = Directive::d_invalid; /* first thing should be a directive */
    nbparam = nullptr;              /* The temporary non-bonded matrix */
    pair    = nullptr;              /* The temporary pair interaction matrix */
    std::vector<std::vector<gmx::ExclusionBlock>> exclusionBlocks;
    nb_funct = F_LJ;

    *reppow = 12.0; /* Default value for repulsion power     */

    /* Init the number of CMAP torsion angles  and grid spacing */
    interactions[F_CMAP].cmakeGridSpacing = 0;
    interactions[F_CMAP].cmapAngles       = 0;

    bWarn_copy_A_B = bFEP;

    PreprocessingBondAtomType bondAtomType;
    /* parse the actual file */
    bReadDefaults = FALSE;
    bGenPairs     = FALSE;
    bReadMolType  = FALSE;
    nmol_couple   = 0;

    do
    {
        status = cpp_read_line(&handle, STRLEN, line);
        done   = (status == eCPP_EOF);
        if (!done)
        {
            if (status != eCPP_OK)
            {
                gmx_fatal(FARGS, "%s", cpp_error(&handle, status));
            }
            else if (out)
            {
                fprintf(out, "%s\n", line);
            }

            set_warning_line(wi, cpp_cur_file(&handle), cpp_cur_linenr(&handle));

            pline = gmx_strdup(line);

            /* Strip trailing '\' from pline, if it exists */
            sl = strlen(pline);
            if ((sl > 0) && (pline[sl - 1] == CONTINUE))
            {
                pline[sl - 1] = ' ';
            }

            /* build one long line from several fragments - necessary for CMAP */
            while (continuing(line))
            {
                status = cpp_read_line(&handle, STRLEN, line);
                set_warning_line(wi, cpp_cur_file(&handle), cpp_cur_linenr(&handle));

                /* Since we depend on the '\' being present to continue to read, we copy line
                 * to a tmp string, strip the '\' from that string, and cat it to pline
                 */
                tmp_line = gmx_strdup(line);

                sl = strlen(tmp_line);
                if ((sl > 0) && (tmp_line[sl - 1] == CONTINUE))
                {
                    tmp_line[sl - 1] = ' ';
                }

                done = (status == eCPP_EOF);
                if (!done)
                {
                    if (status != eCPP_OK)
                    {
                        gmx_fatal(FARGS, "%s", cpp_error(&handle, status));
                    }
                    else if (out)
                    {
                        fprintf(out, "%s\n", line);
                    }
                }

                srenew(pline, strlen(pline) + strlen(tmp_line) + 1);
                strcat(pline, tmp_line);
                sfree(tmp_line);
            }

            /* skip trailing and leading spaces and comment text */
            strip_comment(pline);
            trim(pline);

            /* if there is something left... */
            if (static_cast<int>(strlen(pline)) > 0)
            {
                if (pline[0] == OPENDIR)
                {
                    /* A directive on this line: copy the directive
                     * without the brackets into dirstr, then
                     * skip spaces and tabs on either side of directive
                     */
                    dirstr = gmx_strdup((pline + 1));
                    if ((dummy2 = strchr(dirstr, CLOSEDIR)) != nullptr)
                    {
                        (*dummy2) = 0;
                    }
                    trim(dirstr);

                    if ((newd = str2dir(dirstr)) == Directive::d_invalid)
                    {
                        sprintf(errbuf, "Invalid directive %s", dirstr);
                        warning_error(wi, errbuf);
                    }
                    else
                    {
                        /* Directive found */
                        if (DS_Check_Order(DS, newd))
                        {
                            DS_Push(&DS, newd);
                            d = newd;
                        }
                        else
                        {
                            /* we should print here which directives should have
                               been present, and which actually are */
                            gmx_fatal(FARGS, "%s\nInvalid order for directive %s",
                                      cpp_error(&handle, eCPP_SYNTAX), dir2str(newd));
                            /* d = Directive::d_invalid; */
                        }

                        if (d == Directive::d_intermolecular_interactions)
                        {
                            if (*intermolecular_interactions == nullptr)
                            {
                                /* We (mis)use the moleculetype processing
                                 * to process the intermolecular interactions
                                 * by making a "molecule" of the size of the system.
                                 */
                                *intermolecular_interactions = std::make_unique<MoleculeInformation>();
                                mi0                          = intermolecular_interactions->get();
                                mi0->initMolInfo();
                                make_atoms_sys(*molblock, *molinfo, &mi0->atoms);
                            }
                        }
                    }
                    sfree(dirstr);
                }
                else if (d != Directive::d_invalid)
                {
                    /* Not a directive, just a plain string
                     * use a gigantic switch to decode,
                     * if there is a valid directive!
                     */
                    switch (d)
                    {
                        case Directive::d_defaults:
                            if (bReadDefaults)
                            {
                                gmx_fatal(FARGS, "%s\nFound a second defaults directive.\n",
                                          cpp_error(&handle, eCPP_SYNTAX));
                            }
                            bReadDefaults = TRUE;
                            nscan = sscanf(pline, "%s%s%s%lf%lf%lf", nb_str, comb_str, genpairs,
                                           &fLJ, &fQQ, &fPOW);
                            if (nscan < 2)
                            {
                                too_few(wi);
                            }
                            else
                            {
                                bGenPairs = FALSE;
                                fudgeLJ   = 1.0;
                                *fudgeQQ  = 1.0;

                                get_nbparm(nb_str, comb_str, &nb_funct, combination_rule, wi);
                                if (nscan >= 3)
                                {
                                    bGenPairs = (gmx::equalCaseInsensitive(genpairs, "Y", 1));
                                    if (nb_funct != eNBF_LJ && bGenPairs)
                                    {
                                        gmx_fatal(FARGS,
                                                  "Generating pair parameters is only supported "
                                                  "with LJ non-bonded interactions");
                                    }
                                }
                                if (nscan >= 4)
                                {
                                    fudgeLJ = fLJ;
                                }
                                if (nscan >= 5)
                                {
                                    *fudgeQQ = fQQ;
                                }
                                if (nscan >= 6)
                                {
                                    *reppow = fPOW;
                                }
                            }
                            nb_funct = ifunc_index(Directive::d_nonbond_params, nb_funct);

                            break;
                        case Directive::d_atomtypes:
                            push_at(symtab, atypes, &bondAtomType, pline, nb_funct, &nbparam,
                                    bGenPairs ? &pair : nullptr, wi);
                            break;

                        case Directive::d_bondtypes: // Intended to fall through
                        case Directive::d_constrainttypes:
                            push_bt(d, interactions, 2, nullptr, &bondAtomType, pline, wi);
                            break;
                        case Directive::d_pairtypes:
                            if (bGenPairs)
                            {
                                push_nbt(d, pair, atypes, pline, F_LJ14, wi);
                            }
                            else
                            {
                                push_bt(d, interactions, 2, atypes, nullptr, pline, wi);
                            }
                            break;
                        case Directive::d_angletypes:
                            push_bt(d, interactions, 3, nullptr, &bondAtomType, pline, wi);
                            break;
                        case Directive::d_dihedraltypes:
                            /* Special routine that can read both 2 and 4 atom dihedral definitions. */
                            push_dihedraltype(d, interactions, &bondAtomType, pline, wi);
                            break;

                        case Directive::d_nonbond_params:
                            push_nbt(d, nbparam, atypes, pline, nb_funct, wi);
                            break;

                        case Directive::d_implicit_genborn_params: // NOLINT bugprone-branch-clone
                            // Skip this line, so old topologies with
                            // GB parameters can be read.
                            break;

                        case Directive::d_implicit_surface_params:
                            // Skip this line, so that any topologies
                            // with surface parameters can be read
                            // (even though these were never formally
                            // supported).
                            break;

                        case Directive::d_cmaptypes:
                            push_cmaptype(d, interactions, 5, atypes, &bondAtomType, pline, wi);
                            break;

                        case Directive::d_moleculetype:
                        {
                            if (!bReadMolType)
                            {
                                int ntype;
                                if (opts->couple_moltype != nullptr
                                    && (opts->couple_lam0 == ecouplamNONE || opts->couple_lam0 == ecouplamQ
                                        || opts->couple_lam1 == ecouplamNONE
                                        || opts->couple_lam1 == ecouplamQ))
                                {
                                    dcatt = add_atomtype_decoupled(symtab, atypes, &nbparam,
                                                                   bGenPairs ? &pair : nullptr);
                                }
                                ntype  = atypes->size();
                                ncombs = (ntype * (ntype + 1)) / 2;
                                generate_nbparams(*combination_rule, nb_funct,
                                                  &(interactions[nb_funct]), atypes, wi);
                                ncopy = copy_nbparams(nbparam, nb_funct, &(interactions[nb_funct]), ntype);
                                GMX_LOG(logger.info)
                                        .asParagraph()
                                        .appendTextFormatted(
                                                "Generated %d of the %d non-bonded parameter "
                                                "combinations",
                                                ncombs - ncopy, ncombs);
                                free_nbparam(nbparam, ntype);
                                if (bGenPairs)
                                {
                                    gen_pairs((interactions[nb_funct]), &(interactions[F_LJ14]),
                                              fudgeLJ, *combination_rule);
                                    ncopy = copy_nbparams(pair, nb_funct, &(interactions[F_LJ14]), ntype);
                                    GMX_LOG(logger.info)
                                            .asParagraph()
                                            .appendTextFormatted(
                                                    "Generated %d of the %d 1-4 parameter "
                                                    "combinations",
                                                    ncombs - ncopy, ncombs);
                                    free_nbparam(pair, ntype);
                                }
                                /* Copy GBSA parameters to atomtype array? */

                                bReadMolType = TRUE;
                            }

                            push_molt(symtab, molinfo, pline, wi);
                            exclusionBlocks.emplace_back();
                            mi0                    = &molinfo->back();
                            mi0->atoms.haveMass    = TRUE;
                            mi0->atoms.haveCharge  = TRUE;
                            mi0->atoms.haveType    = TRUE;
                            mi0->atoms.haveBState  = TRUE;
                            mi0->atoms.havePdbInfo = FALSE;
                            break;
                        }
                        case Directive::d_atoms:
                            push_atom(symtab, &(mi0->atoms), atypes, pline, wi);
                            break;

                        case Directive::d_pairs:
                            GMX_RELEASE_ASSERT(
                                    mi0,
                                    "Need to have a valid MoleculeInformation object to work on");
                            push_bond(d, interactions, mi0->interactions, &(mi0->atoms), atypes,
                                      pline, FALSE, bGenPairs, *fudgeQQ, bZero, &bWarn_copy_A_B, wi);
                            break;
                        case Directive::d_pairs_nb:
                            GMX_RELEASE_ASSERT(
                                    mi0,
                                    "Need to have a valid MoleculeInformation object to work on");
                            push_bond(d, interactions, mi0->interactions, &(mi0->atoms), atypes,
                                      pline, FALSE, FALSE, 1.0, bZero, &bWarn_copy_A_B, wi);
                            break;

                        case Directive::d_vsites1:
                        case Directive::d_vsites2:
                        case Directive::d_vsites3:
                        case Directive::d_vsites4:
                        case Directive::d_bonds:
                        case Directive::d_angles:
                        case Directive::d_constraints:
                        case Directive::d_settles:
                        case Directive::d_position_restraints:
                        case Directive::d_angle_restraints:
                        case Directive::d_angle_restraints_z:
                        case Directive::d_distance_restraints:
                        case Directive::d_orientation_restraints:
                        case Directive::d_dihedral_restraints:
                        case Directive::d_dihedrals:
                        case Directive::d_polarization:
                        case Directive::d_water_polarization:
                        case Directive::d_thole_polarization:
                            GMX_RELEASE_ASSERT(
                                    mi0,
                                    "Need to have a valid MoleculeInformation object to work on");
                            push_bond(d, interactions, mi0->interactions, &(mi0->atoms), atypes,
                                      pline, TRUE, bGenPairs, *fudgeQQ, bZero, &bWarn_copy_A_B, wi);
                            break;
                        case Directive::d_cmap:
                            GMX_RELEASE_ASSERT(
                                    mi0,
                                    "Need to have a valid MoleculeInformation object to work on");
                            push_cmap(d, interactions, mi0->interactions, &(mi0->atoms), atypes, pline, wi);
                            break;

                        case Directive::d_vsitesn:
                            GMX_RELEASE_ASSERT(
                                    mi0,
                                    "Need to have a valid MoleculeInformation object to work on");
                            push_vsitesn(d, mi0->interactions, &(mi0->atoms), pline, wi);
                            break;
                        case Directive::d_exclusions:
                            GMX_ASSERT(!exclusionBlocks.empty(),
                                       "exclusionBlocks must always be allocated so exclusions can "
                                       "be processed");
                            if (exclusionBlocks.back().empty())
                            {
                                GMX_RELEASE_ASSERT(mi0,
                                                   "Need to have a valid MoleculeInformation "
                                                   "object to work on");
                                exclusionBlocks.back().resize(mi0->atoms.nr);
                            }
                            push_excl(pline, exclusionBlocks.back(), wi);
                            break;
                        case Directive::d_system:
                            trim(pline);
                            title = put_symtab(symtab, pline);
                            break;
                        case Directive::d_molecules:
                        {
                            int  whichmol;
                            bool bCouple;

                            push_mol(*molinfo, pline, &whichmol, &nrcopies, wi);
                            mi0 = &((*molinfo)[whichmol]);
                            molblock->resize(molblock->size() + 1);
                            molblock->back().type = whichmol;
                            molblock->back().nmol = nrcopies;

                            bCouple = (opts->couple_moltype != nullptr
                                       && (gmx_strcasecmp("system", opts->couple_moltype) == 0
                                           || strcmp(*(mi0->name), opts->couple_moltype) == 0));
                            if (bCouple)
                            {
                                nmol_couple += nrcopies;
                            }

                            if (mi0->atoms.nr == 0)
                            {
                                gmx_fatal(FARGS, "Molecule type '%s' contains no atoms", *mi0->name);
                            }
                            GMX_LOG(logger.info)
                                    .asParagraph()
                                    .appendTextFormatted(
                                            "Excluding %d bonded neighbours molecule type '%s'",
                                            mi0->nrexcl, *mi0->name);
                            sum_q(&mi0->atoms, nrcopies, &qt, &qBt);
                            if (!mi0->bProcessed)
                            {
                                generate_excl(mi0->nrexcl, mi0->atoms.nr, mi0->interactions, &(mi0->excls));
                                gmx::mergeExclusions(&(mi0->excls), exclusionBlocks[whichmol]);
                                make_shake(mi0->interactions, &mi0->atoms, opts->nshake, logger);

                                if (bCouple)
                                {
                                    convert_moltype_couple(mi0, dcatt, *fudgeQQ, opts->couple_lam0,
                                                           opts->couple_lam1, opts->bCoupleIntra,
                                                           nb_funct, &(interactions[nb_funct]), wi);
                                }
                                stupid_fill_block(&mi0->mols, mi0->atoms.nr, TRUE);
                                mi0->bProcessed = TRUE;
                            }
                            break;
                        }
                        default:
                            GMX_LOG(logger.warning)
                                    .asParagraph()
                                    .appendTextFormatted("case: %d", static_cast<int>(d));
                            gmx_incons("unknown directive");
                    }
                }
            }
            sfree(pline);
            pline = nullptr;
        }
    } while (!done);

    // Check that all strings defined with -D were used when processing topology
    std::string unusedDefineWarning = checkAndWarnForUnusedDefines(*handle);
    if (!unusedDefineWarning.empty())
    {
        warning(wi, unusedDefineWarning);
    }

    sfree(cpp_opts_return);

    if (out)
    {
        gmx_fio_fclose(out);
    }

    /* List of GROMACS define names for force fields that have been
     * parametrized using constraints involving hydrogens only.
     *
     * We should avoid hardcoded names, but this is hopefully only
     * needed temparorily for discouraging use of constraints=all-bonds.
     */
    const std::array<std::string, 3> ffDefines = { "_FF_AMBER", "_FF_CHARMM", "_FF_OPLSAA" };
    *ffParametrizedWithHBondConstraints        = false;
    for (const std::string& ffDefine : ffDefines)
    {
        if (cpp_find_define(&handle, ffDefine))
        {
            *ffParametrizedWithHBondConstraints = true;
        }
    }

    if (cpp_find_define(&handle, "_FF_GROMOS96") != nullptr)
    {
        warning(wi,
                "The GROMOS force fields have been parametrized with a physically incorrect "
                "multiple-time-stepping scheme for a twin-range cut-off. When used with "
                "a single-range cut-off (or a correct Trotter multiple-time-stepping scheme), "
                "physical properties, such as the density, might differ from the intended values. "
                "Since there are researchers actively working on validating GROMOS with modern "
                "integrators we have not yet removed the GROMOS force fields, but you should be "
                "aware of these issues and check if molecules in your system are affected before "
                "proceeding. "
                "Further information is available at https://redmine.gromacs.org/issues/2884 , "
                "and a longer explanation of our decision to remove physically incorrect "
                "algorithms "
                "can be found at https://doi.org/10.26434/chemrxiv.11474583.v1 .");
    }
    // TODO: Update URL for Issue #2884 in conjunction with updating grompp.warn in regressiontests.

    cpp_done(handle);

    if (opts->couple_moltype)
    {
        if (nmol_couple == 0)
        {
            gmx_fatal(FARGS, "Did not find any molecules of type '%s' for coupling", opts->couple_moltype);
        }
        GMX_LOG(logger.info)
                .asParagraph()
                .appendTextFormatted("Coupling %d copies of molecule type '%s'", nmol_couple,
                                     opts->couple_moltype);
    }

    /* this is not very clean, but fixes core dump on empty system name */
    if (!title)
    {
        title = put_symtab(symtab, "");
    }

    if (fabs(qt) > 1e-4)
    {
        sprintf(warn_buf, "System has non-zero total charge: %.6f\n%s\n", qt, floating_point_arithmetic_tip);
        warning_note(wi, warn_buf);
    }
    if (fabs(qBt) > 1e-4 && !gmx_within_tol(qBt, qt, 1e-6))
    {
        sprintf(warn_buf, "State B has non-zero total charge: %.6f\n%s\n", qBt,
                floating_point_arithmetic_tip);
        warning_note(wi, warn_buf);
    }
    if (usingFullRangeElectrostatics && (fabs(qt) > 1e-4 || fabs(qBt) > 1e-4))
    {
        warning(wi,
                "You are using Ewald electrostatics in a system with net charge. This can lead to "
                "severe artifacts, such as ions moving into regions with low dielectric, due to "
                "the uniform background charge. We suggest to neutralize your system with counter "
                "ions, possibly in combination with a physiological salt concentration.");
        please_cite(stdout, "Hub2014a");
    }

    DS_Done(&DS);

    if (*intermolecular_interactions != nullptr)
    {
        sfree(intermolecular_interactions->get()->atoms.atom);
    }

    return title;
}

char** do_top(bool                                  bVerbose,
              const char*                           topfile,
              const char*                           topppfile,
              t_gromppopts*                         opts,
              bool                                  bZero,
              t_symtab*                             symtab,
              gmx::ArrayRef<InteractionsOfType>     interactions,
              int*                                  combination_rule,
              double*                               repulsion_power,
              real*                                 fudgeQQ,
              PreprocessingAtomTypes*               atypes,
              std::vector<MoleculeInformation>*     molinfo,
              std::unique_ptr<MoleculeInformation>* intermolecular_interactions,
              const t_inputrec*                     ir,
              std::vector<gmx_molblock_t>*          molblock,
              bool*                                 ffParametrizedWithHBondConstraints,
              warninp*                              wi,
              const gmx::MDLogger&                  logger)
{
    /* Tmpfile might contain a long path */
    const char* tmpfile;
    char**      title;

    if (topppfile)
    {
        tmpfile = topppfile;
    }
    else
    {
        tmpfile = nullptr;
    }

    if (bVerbose)
    {
        GMX_LOG(logger.info).asParagraph().appendTextFormatted("processing topology...");
    }
    title = read_topol(topfile, tmpfile, opts->define, opts->include, symtab, atypes, molinfo,
                       intermolecular_interactions, interactions, combination_rule, repulsion_power,
                       opts, fudgeQQ, molblock, ffParametrizedWithHBondConstraints,
                       ir->efep != efepNO, bZero, EEL_FULL(ir->coulombtype), wi, logger);

    if ((*combination_rule != eCOMB_GEOMETRIC) && (ir->vdwtype == evdwUSER))
    {
        warning(wi,
                "Using sigma/epsilon based combination rules with"
                " user supplied potential function may produce unwanted"
                " results");
    }

    return title;
}

/*! \brief Bookkeeping of the force-field terms dropped by generate_qmexcl_moltype().
 *
 * Counted per interaction function type, and split by how the atoms of the term
 * are distributed over the two regions, so that the effect of the chosen scheme
 * on the QM/MM boundary can be read off the grompp output.
 */
struct QmmmRemovedInteractions
{
    //! Terms all of whose atoms are QM -- described by the QM calculation itself.
    std::array<int, F_NRE> allQm = {};
    //! Terms with atoms in both regions -- these are the QM/MM boundary terms.
    std::array<int, F_NRE> boundary = {};
    //! Terms without a single QM atom (only single-atom terms can end up here).
    std::array<int, F_NRE> mmOnly = {};
};

/*! \brief Detailed, per-atom record of what generate_qmexcl_moltype() changed.
 *
 * All atom numbers are global and 0-based here; they are written 1-based, i.e. in
 * the numbering of the input coordinate file. The labels are stored together with
 * the numbers, because the molecule types are modified while the report is filled.
 */
struct QmmmTopologyReport
{
    //! One atom of a reported term: global index, "RESnr NAME" label, QM or not
    struct Atom
    {
        int         index;
        std::string label;
        bool        isQm;
    };
    //! A force-field term (bonded interaction or pair) with its atoms
    struct Term
    {
        int               ftype;
        std::vector<Atom> atoms;
    };
    //! A bond with exactly one QM atom, and whether it makes its MM atom a boundary atom
    struct BoundaryBond
    {
        int         ftype;
        Atom        qm;
        Atom        mm;
        bool        accepted;
        std::string reason;
    };
    //! A nonbonded (LJ) exclusion between a QM atom and another atom
    struct Exclusion
    {
        Atom qm;
        Atom other;
        int  bondDistance; //!< number of bonds between the two atoms, -1 if more than 7
        bool presentBefore; //!< already excluded by the force field (nrexcl) before QM/MM
    };
    std::vector<BoundaryBond> boundaryBonds;
    std::vector<Term>         removedBonded;
    std::vector<Term>         convertedBonds;
    std::vector<Term>         removedPairs;
    std::vector<Exclusion>    qmQmExclusions;
    std::vector<Exclusion>    qmMmExclusions;
    std::vector<Atom>         qmAtoms;
    //! QM--MM exclusions of the final topology; presentBefore marks a pair with a link atom
    std::vector<Exclusion> finalQmMmExclusions;
    //! QM--MM LJ-14 pairs kept in the final topology
    std::vector<Exclusion> keptQmMmPairs;
    //! Whether the LJ and LJ-14 interactions of the boundary MM atoms with all QM atoms are excluded
    bool excludeBoundaryLJ = false;
};

//! "RESnr NAME" label of a local atom of a molecule type
static std::string qmmmAtomLabel(const gmx_moltype_t& molt, int localIndex)
{
    const t_atoms& atoms = molt.atoms;
    const t_resinfo& ri  = atoms.resinfo[atoms.atom[localIndex].resind];
    return gmx::formatString("%s%d %s", *ri.name, ri.nr, *atoms.atomname[localIndex]);
}

//! Bond distances (up to \p maxDepth) from local atom \p start, over the chemical bonds
static std::vector<int> qmmmBondDistances(const std::vector<std::vector<int>>& graph, int start, int maxDepth)
{
    std::vector<int> depth(graph.size(), -1);
    std::vector<int> frontier{ start };
    depth[start] = 0;
    for (int d = 1; d <= maxDepth && !frontier.empty(); d++)
    {
        std::vector<int> next;
        for (int a : frontier)
        {
            for (int b : graph[a])
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
}

/*! \brief Whether \p ftype is a restraint set by the user rather than a force-field term.
 *
 * Restraints are never described by the QM calculation, so they are kept at any
 * QM/MM boundary and inside the QM region, with every scheme.
 */
static bool isQmmmKeptRestraint(int ftype)
{
    switch (ftype)
    {
        case F_RESTRBONDS:
        case F_POSRES:
        case F_FBPOSRES:
        case F_DISRES:
        case F_ORIRES:
        case F_ANGRES:
        case F_ANGRESZ:
        case F_DIHRES: return true;
        default: return false;
    }
}

/*! \brief
 * Exclude molecular interactions for QM atoms in QM/MM
 *
 * Update the exclusion lists to include all QM atoms of this molecule,
 * replace bonds between QM atoms with CONNBOND and
 * set charges of QM atoms to 0.
 * With the classic scheme, also remove the interactions between the QM atoms
 * and the link atoms.
 *
 * The bonded interactions that involve both QM and MM atoms are treated
 * according to one of two conventions, selected with \p qmmmMode:
 *
 * GMX_QMMM_ORIGINAL ("classic", the default and the historical GROMACS
 *   behaviour): a bonded interaction is removed as soon as all but one of its
 *   atoms are QM (i.e. a QM-QM-MM angle and a QM-QM-QM-MM dihedral are removed),
 *   because the QM calculation with the link atom is assumed to describe it
 *   already, and keeping the force-field term would count it twice.
 *   The LJ between QM and MM atoms follows the exclusion rules of the force
 *   field (nrexcl and [ pairs ]). Only with \p excludeBoundaryLJ, as in earlier
 *   versions, the nonbonded and 1-4 interactions of every QM atom with the MM
 *   atoms covalently bound to the QM region are excluded as well.
 *
 * GMX_QMMM_AMBER ("amber"): only those interactions are removed whose atoms are
 *   all QM; every term with at least one MM atom is kept at the force-field
 *   level, as in the QM/MM implementation of AMBER. The reasoning is that the
 *   link atom is not the MM atom -- it differs in position and in mass -- so its
 *   contribution is not equivalent to the force-field term, and the conformational
 *   behaviour of the boundary is better described by keeping the latter.
 *   Consistently with that, no exclusions involving link atoms are generated.
 *
 * GMX_QMMM_MIMIC uses the same rule for the bonded interactions as
 *   GMX_QMMM_AMBER, because MiMiC treats the link atoms as quantum atoms.
 *
 * Restraints (position, flat-bottomed position, distance, orientation, angle,
 *   dihedral restraints and restraint potentials) are kept with every scheme,
 *   also on QM atoms: they are set by the user and not part of the force field
 *   that the QM calculation replaces.
 *
 * \param[in,out] molt molecule type with QM atoms
 * \param[in] grpnr group informatio
 * \param[in,out] ir input record
 * \param[in,out] qmmmMode QM/MM mode switch: original(classic)/MiMiC/amber
 * \param[in] logger Handle to logging interface.
 * \param[in,out] removed Counters of the removed interactions, accumulated over
 *                        all of the molecule types that contain QM atoms.
 * \param[in] atomOffset Global index of the first atom of this molecule, for the report.
 * \param[in,out] report Per-atom record of the changes, written to a file at the end.
 * \param[in] excludeBoundaryLJ Exclude the LJ and LJ-14 of the boundary MM atoms with
 *                              every QM atom (classic scheme only, GMX_QMMM_LJ_SCHEME=exclude).
 */
static void generate_qmexcl_moltype(gmx_moltype_t*          molt,
                                    const unsigned char*    grpnr,
                                    t_inputrec*             ir,
                                    GmxQmmmMode             qmmmMode,
                                    const gmx::MDLogger&    logger,
                                    QmmmRemovedInteractions* removed,
                                    int                     atomOffset,
                                    QmmmTopologyReport*     report,
                                    bool                    excludeBoundaryLJ)
{
    /* This routine expects molt->ilist to be of size F_NRE and ordered. */

    /* generates the exclusions between the individual QM atoms, as
     * these interactions should be handled by the QM subroutines and
     * not by the gromacs routines
     */
    int   qm_max = 0, qm_nr = 0, link_nr = 0, link_max = 0;
    int * qm_arr = nullptr, *link_arr = nullptr;
    bool *bQMMM, *blink;

    /* First we search and select the QM atoms in an qm_arr array that
     * we use to create the exclusions.
     *
     * we take the possibility into account that a user has defined more
     * than one QM group:
     *
     * for that we also need to do this an ugly work-about just in case
     * the QM group contains the entire system...
     */

    /* we first search for all the QM atoms and put them in an array
     */
    for (int j = 0; j < ir->opts.ngQM; j++)
    {
        for (int i = 0; i < molt->atoms.nr; i++)
        {
            if (qm_nr >= qm_max)
            {
                qm_max += 100;
                srenew(qm_arr, qm_max);
            }
            if ((grpnr ? grpnr[i] : 0) == j)
            {
                qm_arr[qm_nr++]        = i;
                molt->atoms.atom[i].q  = 0.0;
                molt->atoms.atom[i].qB = 0.0;
            }
        }
    }
    /* bQMMM[..] is an array containin TRUE/FALSE for atoms that are
     * QM/not QM. We first set all elements to false. Afterwards we use
     * the qm_arr to change the elements corresponding to the QM atoms
     * to TRUE.
     */
    snew(bQMMM, molt->atoms.nr);
    for (int i = 0; i < molt->atoms.nr; i++)
    {
        bQMMM[i] = FALSE;
    }
    for (int i = 0; i < qm_nr; i++)
    {
        bQMMM[qm_arr[i]] = TRUE;
    }

    const auto reportAtom = [&](int local) {
        return QmmmTopologyReport::Atom{ atomOffset + local, qmmmAtomLabel(*molt, local),
                                         static_cast<bool>(bQMMM[local]) };
    };
    const auto reportTerm = [&](int ftype, const int* iatoms, int nratoms) {
        QmmmTopologyReport::Term term{ ftype, {} };
        for (int k = 0; k < nratoms; k++)
        {
            term.atoms.push_back(reportAtom(iatoms[k]));
        }
        return term;
    };
    for (int i = 0; i < qm_nr; i++)
    {
        report->qmAtoms.push_back(reportAtom(qm_arr[i]));
    }

    /* The atoms each virtual site is constructed from. A link atom is a virtual
     * site in the QM group, and the bonds that start from it are connections
     * (funct 5) that only serve to generate exclusions.
     */
    std::vector<std::vector<int>> vsiteConstructingAtoms(molt->atoms.nr);
    for (int ftype = 0; ftype < F_NRE; ftype++)
    {
        if (!(interaction_function[ftype].flags & IF_VSITE))
        {
            continue;
        }
        const int              nratoms = interaction_function[ftype].nratoms;
        const InteractionList& il      = molt->ilist[ftype];
        for (int i = 0; i < il.size(); i += 1 + nratoms)
        {
            for (int k = 2; k <= nratoms; k++)
            {
                vsiteConstructingAtoms[il.iatoms[i + 1]].push_back(il.iatoms[i + k]);
            }
        }
    }

    /* Chemical-bond graph of the molecule, for the bond distances in the report */
    std::vector<std::vector<int>> bondGraph(molt->atoms.nr);
    for (int ftype = 0; ftype < F_NRE; ftype++)
    {
        if (IS_CHEMBOND(ftype))
        {
            const InteractionList& il = molt->ilist[ftype];
            for (int i = 0; i < il.size(); i += 3)
            {
                bondGraph[il.iatoms[i + 1]].push_back(il.iatoms[i + 2]);
                bondGraph[il.iatoms[i + 2]].push_back(il.iatoms[i + 1]);
            }
        }
    }

    /* We remove all bonded interactions (i.e. bonds,
     * angles, dihedrals, 1-4's), involving the QM atoms. The way they
     * are removed is as follows: if the interaction invloves 2 atoms,
     * it is removed if both atoms are QMatoms. If it involves 3 atoms,
     * it is removed if at least two of the atoms are QM atoms, if the
     * interaction involves 4 atoms, it is removed if there are at least
     * 2 QM atoms.  Since this routine is called once before any forces
     * are computed, the top->idef.il[N].iatom[] array (see idef.h) can
     * be rewritten at this poitn without any problem. 25-9-2002 */

    /* first check whether we already have CONNBONDS.
     * Note that if we don't, we don't add a param entry and set ftype=0,
     * which is ok, since CONNBONDS does not use parameters.
     */
    int ftype_connbond = 0;
    int ind_connbond   = 0;
    if (!molt->ilist[F_CONNBONDS].empty())
    {
        GMX_LOG(logger.info)
                .asParagraph()
                .appendTextFormatted("nr. of CONNBONDS present already: %d",
                                     molt->ilist[F_CONNBONDS].size() / 3);
        ftype_connbond = molt->ilist[F_CONNBONDS].iatoms[0];
        ind_connbond   = molt->ilist[F_CONNBONDS].size();
    }
    /* now we delete all bonded interactions, except the ones describing
     * a chemical bond. These are converted to CONNBONDS
     */
    for (int ftype = 0; ftype < F_NRE; ftype++)
    {
        if (!(interaction_function[ftype].flags & IF_BOND) || ftype == F_CONNBONDS
            || isQmmmKeptRestraint(ftype))
        {
            continue;
        }
        int nratoms = interaction_function[ftype].nratoms;
        int j       = 0;
        while (j < molt->ilist[ftype].size())
        {
            bool bexcl;

            /* How many of the atoms of this interaction are QM? Only needed for
             * the bookkeeping below, and for the multi-atom interactions.
             */
            int numQmAtoms = 0;
            for (int jj = j + 1; jj < j + 1 + nratoms; jj++)
            {
                if (bQMMM[molt->ilist[ftype].iatoms[jj]])
                {
                    numQmAtoms++;
                }
            }

            if (nratoms == 2)
            {
                /* Remove an interaction between two atoms when both are
                 * in the QM region. Note that we don't have to worry about
                 * link atoms here, as they won't have 2-atom interactions.
                 */
                int a1 = molt->ilist[ftype].iatoms[1 + j + 0];
                int a2 = molt->ilist[ftype].iatoms[1 + j + 1];
                bexcl  = (bQMMM[a1] && bQMMM[a2]);
                /* A chemical bond between two QM atoms will be copied to
                 * the F_CONNBONDS list, for reasons mentioned above.
                 */
                if (bexcl && IS_CHEMBOND(ftype))
                {
                    report->convertedBonds.push_back(
                            reportTerm(ftype, molt->ilist[ftype].iatoms.data() + j + 1, nratoms));
                    InteractionList& ilist = molt->ilist[F_CONNBONDS];
                    ilist.iatoms.resize(ind_connbond + 3);
                    ilist.iatoms[ind_connbond++] = ftype_connbond;
                    ilist.iatoms[ind_connbond++] = a1;
                    ilist.iatoms[ind_connbond++] = a2;
                }
            }
            else
            {
                /* MM interactions have to be excluded if they are included
                 * in the QM already. Because we use a link atom (H atom)
                 * when the QM/MM boundary runs through a chemical bond, this
                 * means that as long as one atom is MM, we still exclude,
                 * as the interaction is included in the QM via:
                 * QMatom1-QMatom2-QMatom-3-Linkatom.
                 */

                /* MiMiC treats link atoms as quantum atoms - therefore
                 * we do not need do additional exclusions here.
                 * The "amber" scheme uses the same rule for a different reason:
                 * an interaction is only removed if it is described by the QM
                 * calculation completely, i.e. if all of its atoms are QM.
                 * (Restraints, including the single-atom position restraints,
                 * never get here: they are kept with every scheme.)
                 */
                if (qmmmMode == GmxQmmmMode::GMX_QMMM_MIMIC || qmmmMode == GmxQmmmMode::GMX_QMMM_AMBER)
                {
                    bexcl = numQmAtoms == nratoms;
                }
                else
                {
                    bexcl = (numQmAtoms >= nratoms - 1);
                }

                if (bexcl && ftype == F_SETTLE)
                {
                    gmx_fatal(FARGS,
                              "Can not apply QM to molecules with SETTLE, replace the moleculetype "
                              "using QM and SETTLE by one without SETTLE");
                }
            }
            if (bexcl)
            {
                /* keep track of what is being removed, for the report at the end */
                if (interaction_function[ftype].flags & IF_PAIR)
                {
                    /* pairs of two QM atoms are removed here already; list them
                     * with the other removed pairs */
                    report->removedPairs.push_back(
                            reportTerm(ftype, molt->ilist[ftype].iatoms.data() + j + 1, nratoms));
                }
                else if (!(nratoms == 2 && IS_CHEMBOND(ftype)))
                {
                    report->removedBonded.push_back(
                            reportTerm(ftype, molt->ilist[ftype].iatoms.data() + j + 1, nratoms));
                }
                if (numQmAtoms == nratoms)
                {
                    removed->allQm[ftype]++;
                }
                else if (numQmAtoms > 0)
                {
                    removed->boundary[ftype]++;
                }
                else
                {
                    removed->mmOnly[ftype]++;
                }
                /* since the interaction involves QM atoms, these should be
                 * removed from the MM ilist
                 */
                InteractionList& ilist = molt->ilist[ftype];
                for (int k = j; k < ilist.size() - (nratoms + 1); k++)
                {
                    ilist.iatoms[k] = ilist.iatoms[k + (nratoms + 1)];
                }
                ilist.iatoms.resize(ilist.size() - (nratoms + 1));
            }
            else
            {
                j += nratoms + 1; /* the +1 is for the functype */
            }
        }
    }
    /* Now, we search for atoms bonded to a QM atom because we also want
     * to exclude their nonbonded interactions with the QM atoms. The
     * reason for this is that this interaction is accounted for in the
     * linkatoms interaction with the QMatoms and would be counted
     * twice.
     * This is only done with the classic scheme. The amber scheme keeps the
     * interactions of the link atoms with the QM atoms at the force-field level
     * (the ones within the exclusion range of the force field are excluded by
     * the exclusions generated by grompp anyway), and MiMiC does not need it
     * because it treats the link atoms as quantum atoms.
     * If no link atoms are collected here, blink[] remains false everywhere,
     * which keeps the exclusions and the 1-4 pairs below untouched. */

    if (qmmmMode == GmxQmmmMode::GMX_QMMM_ORIGINAL)
    {
        for (int i = 0; i < F_NRE; i++)
        {
            if (IS_CHEMBOND(i))
            {
                int j = 0;
                while (j < molt->ilist[i].size())
                {
                    int a1 = molt->ilist[i].iatoms[j + 1];
                    int a2 = molt->ilist[i].iatoms[j + 2];
                    if ((bQMMM[a1] && !bQMMM[a2]) || (!bQMMM[a1] && bQMMM[a2]))
                    {
                        const int qmAtom = bQMMM[a1] ? a1 : a2;
                        const int mmAtom = bQMMM[a1] ? a2 : a1;
                        /* A bond from a link atom (a virtual site of the QM group) is
                         * a connection that only generates exclusions. It marks a
                         * boundary MM atom only if that atom is one the link atom is
                         * constructed from, i.e. the MM atom of the cut bond. Bonds
                         * from a link atom to the further neighbours (MM2) must not
                         * extend the LJ and LJ-14 exclusions to those atoms.
                         */
                        const std::vector<int>& constructing = vsiteConstructingAtoms[qmAtom];
                        const bool              fromLinkAtom = !constructing.empty();
                        const bool              accepted =
                                !fromLinkAtom
                                || std::find(constructing.begin(), constructing.end(), mmAtom)
                                           != constructing.end();
                        report->boundaryBonds.push_back(
                                { i, reportAtom(qmAtom), reportAtom(mmAtom), accepted,
                                  !fromLinkAtom ? "bond of a QM atom"
                                                : (accepted ? "link atom constructed from this MM atom"
                                                            : "link atom NOT constructed from this MM atom: ignored") });
                        if (accepted)
                        {
                            if (link_nr >= link_max)
                            {
                                link_max += 10;
                                srenew(link_arr, link_max);
                            }
                            link_arr[link_nr++] = mmAtom;
                        }
                    }
                    j += 3;
                }
            }
        }
    }
    snew(blink, molt->atoms.nr);
    for (int i = 0; i < molt->atoms.nr; i++)
    {
        blink[i] = FALSE;
    }

    /* The boundary MM atoms collected above lose their LJ and LJ-14 interactions
     * with every QM atom only on request (GMX_QMMM_LJ_SCHEME=exclude). By default
     * the LJ between the QM and the MM atoms follows the exclusion rules of the force
     * field: the pairs within nrexcl bonds are excluded by grompp anyway, the 1-4 pairs
     * keep their LJ-14, and every other pair keeps its LJ. The collected atoms are
     * still listed in the report.
     */
    const int numLinkExclusions = (qmmmMode == GmxQmmmMode::GMX_QMMM_ORIGINAL && excludeBoundaryLJ)
                                          ? link_nr
                                          : 0;
    for (int i = 0; i < numLinkExclusions; i++)
    {
        blink[link_arr[i]] = TRUE;
    }
    /* creating the exclusion block for the QM atoms. Each QM atom has
     * as excluded elements all the other QMatoms (and itself).
     */
    t_blocka qmexcl;
    qmexcl.nr  = molt->atoms.nr;
    qmexcl.nra = qm_nr * (qm_nr + numLinkExclusions) + numLinkExclusions * qm_nr;
    snew(qmexcl.index, qmexcl.nr + 1);
    snew(qmexcl.a, qmexcl.nra);
    int j = 0;
    for (int i = 0; i < qmexcl.nr; i++)
    {
        qmexcl.index[i] = j;
        if (bQMMM[i])
        {
            for (int k = 0; k < qm_nr; k++)
            {
                qmexcl.a[k + j] = qm_arr[k];
            }
            for (int k = 0; k < numLinkExclusions; k++)
            {
                qmexcl.a[qm_nr + k + j] = link_arr[k];
            }
            j += (qm_nr + numLinkExclusions);
        }
        if (blink[i])
        {
            for (int k = 0; k < qm_nr; k++)
            {
                qmexcl.a[k + j] = qm_arr[k];
            }
            j += qm_nr;
        }
    }
    qmexcl.index[qmexcl.nr] = j;

    /* record the exclusions for the report, and whether the force field
     * (nrexcl) had excluded the pair already
     */
    {
        const auto excludedBefore = [&](int a, int b) {
            const auto list = molt->excls[a];
            return std::find(list.begin(), list.end(), b) != list.end();
        };
        std::vector<bool> reported(molt->atoms.nr, false);
        for (int a = 0; a < qm_nr; a++)
        {
            const int              qa    = qm_arr[a];
            const std::vector<int> depth = qmmmBondDistances(bondGraph, qa, 7);
            for (int b = a + 1; b < qm_nr; b++)
            {
                const int qb = qm_arr[b];
                report->qmQmExclusions.push_back(
                        { reportAtom(qa), reportAtom(qb), depth[qb], excludedBefore(qa, qb) });
            }
            std::fill(reported.begin(), reported.end(), false);
            for (int k = 0; k < numLinkExclusions; k++)
            {
                const int mm = link_arr[k];
                if (reported[mm])
                {
                    continue; // the same MM atom can be reached by more than one bond
                }
                reported[mm] = true;
                report->qmMmExclusions.push_back(
                        { reportAtom(qa), reportAtom(mm), depth[mm], excludedBefore(qa, mm) });
            }
        }
    }

    /* and merging with the exclusions already present in sys.
     */

    std::vector<gmx::ExclusionBlock> qmexcl2(molt->atoms.nr);
    gmx::blockaToExclusionBlocks(&qmexcl, qmexcl2);
    gmx::mergeExclusions(&(molt->excls), qmexcl2);

    /* Finally, we also need to get rid of the pair interactions of the
     * classical atom bonded to the boundary QM atoms with the QMatoms,
     * as this interaction is already accounted for by the QM, so also
     * here we run the risk of double counting! We proceed in a similar
     * way as we did above for the other bonded interactions:
     * (with the amber scheme and with MiMiC, blink[] is false everywhere,
     *  so only the pairs between two QM atoms are removed here) */
    for (int i = F_LJ14; i < F_COUL14; i++)
    {
        int nratoms = interaction_function[i].nratoms;
        int j       = 0;
        while (j < molt->ilist[i].size())
        {
            int  a1 = molt->ilist[i].iatoms[j + 1];
            int  a2 = molt->ilist[i].iatoms[j + 2];
            bool bexcl =
                    ((bQMMM[a1] && bQMMM[a2]) || (blink[a1] && bQMMM[a2]) || (bQMMM[a1] && blink[a2]));
            if (bexcl)
            {
                report->removedPairs.push_back(reportTerm(i, molt->ilist[i].iatoms.data() + j + 1, 2));
                /* keep track of what is being removed, for the report at the end:
                 * a pair of two QM atoms, or a QM atom with a link atom (boundary)
                 */
                if (bQMMM[a1] && bQMMM[a2])
                {
                    removed->allQm[i]++;
                }
                else
                {
                    removed->boundary[i]++;
                }
                /* since the interaction involves QM atoms, these should be
                 * removed from the MM ilist
                 */
                InteractionList& ilist = molt->ilist[i];
                for (int k = j; k < ilist.size() - (nratoms + 1); k++)
                {
                    ilist.iatoms[k] = ilist.iatoms[k + (nratoms + 1)];
                }
                ilist.iatoms.resize(ilist.size() - (nratoms + 1));
            }
            else
            {
                j += nratoms + 1; /* the +1 is for the functype */
            }
        }
    }

    /* the QM--MM exclusions and LJ-14 pairs that end up in the tpr, for the report */
    {
        const auto isLinkAtom = [&](int a) { return !vsiteConstructingAtoms[a].empty(); };
        for (int a = 0; a < molt->atoms.nr; a++)
        {
            if (!bQMMM[a])
            {
                continue;
            }
            const std::vector<int> depth = qmmmBondDistances(bondGraph, a, 7);
            std::vector<int>       partners;
            for (int b : molt->excls[a])
            {
                if (!bQMMM[b])
                {
                    partners.push_back(b);
                }
            }
            std::sort(partners.begin(), partners.end());
            for (int b : partners)
            {
                report->finalQmMmExclusions.push_back(
                        { reportAtom(a), reportAtom(b), depth[b], isLinkAtom(a) });
            }
        }
        const InteractionList& il = molt->ilist[F_LJ14];
        for (int k = 0; k < il.size(); k += 3)
        {
            const int a1 = il.iatoms[k + 1], a2 = il.iatoms[k + 2];
            if (bQMMM[a1] != bQMMM[a2])
            {
                const int qa = bQMMM[a1] ? a1 : a2, mb = bQMMM[a1] ? a2 : a1;
                report->keptQmMmPairs.push_back(
                        { reportAtom(qa), reportAtom(mb), qmmmBondDistances(bondGraph, qa, 7)[mb],
                          isLinkAtom(qa) });
            }
        }
    }
    report->excludeBoundaryLJ = (qmmmMode == GmxQmmmMode::GMX_QMMM_ORIGINAL && excludeBoundaryLJ);

    free(qm_arr);
    free(bQMMM);
    free(link_arr);
    free(blink);
} /* generate_qmexcl */

/*! \brief Report the force-field terms that were removed around the QM region.
 *
 * The interesting column is the middle one: those are the terms that connect the
 * two regions, and they are the ones whose treatment differs between the classic
 * and the amber scheme. The terms with only QM atoms are removed by either
 * scheme, because the QM calculation describes them.
 */
static void reportQmmmRemovedInteractions(const QmmmRemovedInteractions& removed,
                                          GmxQmmmMode                    qmmmMode,
                                          const gmx::MDLogger&           logger)
{
    const char* schemeName = "classic";
    if (qmmmMode == GmxQmmmMode::GMX_QMMM_MIMIC)
    {
        schemeName = "MiMiC";
    }
    else if (qmmmMode == GmxQmmmMode::GMX_QMMM_AMBER)
    {
        schemeName = "amber";
    }

    int totalAllQm = 0, totalBoundary = 0, totalMmOnly = 0;
    for (int ftype = 0; ftype < F_NRE; ftype++)
    {
        totalAllQm += removed.allQm[ftype];
        totalBoundary += removed.boundary[ftype];
        totalMmOnly += removed.mmOnly[ftype];
    }

    GMX_LOG(logger.info)
            .appendTextFormatted(
                    "\nQM/MM: force-field terms removed with the '%s' scheme, by interaction type:",
                    schemeName);
    if (totalAllQm + totalBoundary + totalMmOnly == 0)
    {
        GMX_LOG(logger.info).appendTextFormatted("  (none)\n");
        return;
    }
    GMX_LOG(logger.info)
            .appendTextFormatted("  %-22s %10s %10s %10s", "interaction", "all-QM", "QM--MM", "MM-only");
    for (int ftype = 0; ftype < F_NRE; ftype++)
    {
        if (removed.allQm[ftype] + removed.boundary[ftype] + removed.mmOnly[ftype] == 0)
        {
            continue;
        }
        GMX_LOG(logger.info)
                .appendTextFormatted("  %-22s %10d %10d %10d", interaction_function[ftype].longname,
                                     removed.allQm[ftype], removed.boundary[ftype], removed.mmOnly[ftype]);
    }
    GMX_LOG(logger.info)
            .appendTextFormatted("  %-22s %10d %10d %10d", "total", totalAllQm, totalBoundary, totalMmOnly);
    GMX_LOG(logger.info)
            .appendTextFormatted(
                    "  all-QM  = every atom of the term is QM: described by the QM calculation\n"
                    "  QM--MM  = the term spans the QM/MM boundary\n"
                    "  MM-only = no QM atom in the term at all");
    if (totalBoundary == 0)
    {
        GMX_LOG(logger.info)
                .appendTextFormatted(
                        "No term spanning the boundary was removed: every force-field term with at "
                        "least one MM atom is kept.\n");
    }
    else
    {
        GMX_LOG(logger.info)
                .appendTextFormatted(
                        "The QM--MM terms above are assumed to be described by the QM calculation "
                        "with the link atom;\nswitch the scheme with GMX_QMMM_BONDED_SCHEME to keep "
                        "them at the force-field level.\n");
    }
    GMX_LOG(logger.info)
            .appendTextFormatted(
                    "Note: chemical bonds between two QM atoms are not lost but converted to "
                    "connections (F_CONNBONDS),\nand the removed pair interactions (LJ-14) are "
                    "listed here as well.\n");
}


/*! \brief Whether the per-atom QM/MM report files are written.
 *
 * On by default; GMX_QMMM_REPORTS set to 0, no, off or false switches off the
 * reports of both grompp and mdrun, together with the lines that point to them.
 */
static bool qmmmReportsEnabled()
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

//! Writes the detailed per-atom report of the QM/MM changes to the topology
static void writeQmmmTopologyReport(const QmmmTopologyReport& report, GmxQmmmMode qmmmMode, const char* fileName)
{
    FILE* fp = std::fopen(fileName, "w");
    if (fp == nullptr)
    {
        return;
    }
    const auto atomText = [](const QmmmTopologyReport::Atom& a) {
        return gmx::formatString("%7d %-14s %s", a.index + 1, a.label.c_str(), a.isQm ? "QM" : "MM");
    };
    const auto distanceText = [](int d) {
        return d < 0 ? std::string("> 1-8") : gmx::formatString("1-%d", d + 1);
    };
    const char* schemeName = qmmmMode == GmxQmmmMode::GMX_QMMM_MIMIC
                                     ? "MiMiC"
                                     : (qmmmMode == GmxQmmmMode::GMX_QMMM_AMBER ? "amber" : "classic");

    std::fprintf(fp, "; QM/MM changes to the force-field topology, written by gmx grompp\n");
    std::fprintf(fp, "; scheme for the bonded terms at the QM/MM boundary: %s (GMX_QMMM_BONDED_SCHEME)\n",
                 schemeName);
    std::fprintf(fp, "; LJ between QM and MM atoms: %s (GMX_QMMM_LJ_SCHEME)\n",
                 report.excludeBoundaryLJ ? "exclude -- boundary MM atoms lose LJ and LJ-14 with every QM atom"
                                          : "forcefield -- exclusion rules of the force field (nrexcl, [ pairs ])");
    std::fprintf(fp, "; atom numbers are global and 1-based, i.e. the numbering of the input .gro file\n");
    std::fprintf(fp, "; labels are RESIDUEnumber ATOMNAME from the topology; QM/MM marks the region\n");
    std::fprintf(fp, "; the QM/MM electrostatics at the boundary (GMX_QMMM_POT_SCHEME, GMX_QMMM_GRAD_EXCL) are set by mdrun and\n");
    std::fprintf(fp, ";   listed in its own report, not here\n\n");

    std::fprintf(fp, "[ qm_atoms ]\n; %zu atoms; their charges are set to zero in the tpr\n", report.qmAtoms.size());
    for (const auto& a : report.qmAtoms)
    {
        std::fprintf(fp, "%s\n", atomText(a).c_str());
    }

    std::fprintf(fp, "\n[ boundary_bonds ]\n");
    std::fprintf(fp, "; chemical bonds and connections with exactly one QM atom. The MM atom of an\n");
    std::fprintf(fp, "; accepted bond is a boundary MM atom. Only with GMX_QMMM_LJ_SCHEME=exclude\n");
    std::fprintf(fp, "; its LJ and LJ-14 interactions with every QM atom are excluded; %s.\n",
                 report.excludeBoundaryLJ ? "this is the case here" : "not the case here");
    std::fprintf(fp, "; %-26s %-26s %-12s %s\n", "QM atom", "MM atom", "bond type", "boundary MM atom?");
    for (const auto& b : report.boundaryBonds)
    {
        std::fprintf(fp, "%s %s  %-12s %s (%s)\n", atomText(b.qm).c_str(), atomText(b.mm).c_str(),
                     interaction_function[b.ftype].name, b.accepted ? "yes" : "no", b.reason.c_str());
    }
    if (report.boundaryBonds.empty())
    {
        std::fprintf(fp, "; none\n");
    }

    std::fprintf(fp, "\n[ removed_bonded_terms ]\n");
    std::fprintf(fp, "; force-field terms removed from the topology, per interaction type.\n");
    std::fprintf(fp, "; class: boundary = QM and MM atoms, all-QM = described by the QM calculation,\n");
    std::fprintf(fp, ";        MM-only = no QM atom\n");
    std::fprintf(fp, "; restraints (position, distance, orientation, angle, dihedral) are never removed\n");
    for (int ftype = 0; ftype < F_NRE; ftype++)
    {
        for (const char* cls : { "boundary", "all-QM", "MM-only" })
        {
            std::vector<const QmmmTopologyReport::Term*> terms;
            for (const auto& term : report.removedBonded)
            {
                if (term.ftype != ftype)
                {
                    continue;
                }
                int nQm = 0;
                for (const auto& a : term.atoms)
                {
                    nQm += a.isQm ? 1 : 0;
                }
                const char* c = nQm == static_cast<int>(term.atoms.size()) ? "all-QM"
                                                                             : (nQm > 0 ? "boundary" : "MM-only");
                if (std::strcmp(c, cls) == 0)
                {
                    terms.push_back(&term);
                }
            }
            if (terms.empty())
            {
                continue;
            }
            std::fprintf(fp, "; %s, %s: %zu\n", interaction_function[ftype].longname, cls, terms.size());
            for (const auto* term : terms)
            {
                std::string line = gmx::formatString("  %-8s", cls);
                for (const auto& a : term->atoms)
                {
                    line += " |" + atomText(a);
                }
                std::fprintf(fp, "%s\n", line.c_str());
            }
        }
    }
    if (report.removedBonded.empty())
    {
        std::fprintf(fp, "; none\n");
    }

    std::fprintf(fp, "\n[ qm_qm_bonds_converted_to_connections ]\n");
    std::fprintf(fp, "; chemical bonds between two QM atoms: no force any more, but still used for exclusions\n");
    for (const auto& term : report.convertedBonds)
    {
        std::fprintf(fp, "  %-12s |%s |%s\n", interaction_function[term.ftype].name,
                     atomText(term.atoms[0]).c_str(), atomText(term.atoms[1]).c_str());
    }

    for (const bool boundary : { true, false })
    {
        std::vector<const QmmmTopologyReport::Term*> pairs;
        for (const auto& term : report.removedPairs)
        {
            const bool isBoundary = !(term.atoms[0].isQm && term.atoms[1].isQm);
            if (isBoundary == boundary)
            {
                pairs.push_back(&term);
            }
        }
        std::fprintf(fp, "\n[ removed_lj14_pairs_%s ]\n", boundary ? "qm_boundary_mm" : "qm_qm");
        std::fprintf(fp, "; pair interactions (LJ-14 with its Coulomb-14) removed from the topology: %zu\n", pairs.size());
        if (boundary)
        {
            std::fprintf(fp, "; a QM atom with a boundary MM atom -- counted in the QM calculation via the link atom\n");
        }
        for (const auto* term : pairs)
        {
            std::fprintf(fp, "  %-8s |%s |%s\n", interaction_function[term->ftype].name,
                         atomText(term->atoms[0]).c_str(), atomText(term->atoms[1]).c_str());
        }
    }

    for (const bool boundary : { true, false })
    {
        const auto& list = boundary ? report.qmMmExclusions : report.qmQmExclusions;
        int         nNew = 0;
        for (const auto& e : list)
        {
            nNew += e.presentBefore ? 0 : 1;
        }
        std::fprintf(fp, "\n[ lj_exclusions_%s ]\n", boundary ? "qm_boundary_mm" : "qm_qm");
        std::fprintf(fp, "; nonbonded exclusions generated for QM/MM: %zu pairs, %d of them new, %zu already\n",
                     list.size(), nNew, list.size() - nNew);
        std::fprintf(fp, "; excluded by the force field (nrexcl). An excluded pair has neither LJ nor Coulomb;\n");
        std::fprintf(fp, "; the Coulomb of a QM atom is zero anyway, so what is removed here is the LJ.\n");
        std::fprintf(fp, "; %-26s %-26s %-8s %s\n", "QM atom", boundary ? "boundary MM atom" : "QM atom",
                     "bonds", "status");
        for (const auto& e : list)
        {
            std::fprintf(fp, "%s %s  %-8s %s\n", atomText(e.qm).c_str(), atomText(e.other).c_str(),
                         distanceText(e.bondDistance).c_str(),
                         e.presentBefore ? "already excluded by the force field" : "NEW: LJ removed by QM/MM");
        }
        if (boundary && !report.excludeBoundaryLJ)
        {
            std::fprintf(fp, "; none: the LJ of the boundary MM atoms follows the force field\n");
        }
    }

    for (const bool excl : { true, false })
    {
        const auto& list = excl ? report.finalQmMmExclusions : report.keptQmMmPairs;
        int         nLink = 0;
        for (const auto& e : list)
        {
            nLink += e.presentBefore ? 1 : 0;
        }
        std::fprintf(fp, "\n[ %s ]\n", excl ? "final_lj_exclusions_qm_mm" : "final_lj14_pairs_qm_mm");
        std::fprintf(fp,
                     excl ? "; every QM--MM pair without LJ (nor Coulomb) in the tpr: %zu, %d of them with a link atom\n"
                          : "; every QM--MM pair with LJ-14 in the tpr: %zu, %d of them with a link atom\n",
                     list.size(), nLink);
        std::fprintf(fp, "; a link atom has neither charge nor LJ, so its pairs change nothing\n");
        std::fprintf(fp, "; %-26s %-26s %-8s %s\n", "QM atom", "MM atom", "bonds", "");
        for (const auto& e : list)
        {
            std::fprintf(fp, "%s %s  %-8s %s\n", atomText(e.qm).c_str(), atomText(e.other).c_str(),
                         distanceText(e.bondDistance).c_str(), e.presentBefore ? "link atom" : "");
        }
    }
    std::fclose(fp);
}

void generate_qmexcl(gmx_mtop_t* sys, t_inputrec* ir, warninp* wi, GmxQmmmMode qmmmMode, const gmx::MDLogger& logger)
{
    /* This routine expects molt->molt[m].ilist to be of size F_NRE and ordered.
     */

    unsigned char*  grpnr;
    int             mol, nat_mol, nr_mol_with_qm_atoms = 0;
    gmx_molblock_t* molb;
    bool            bQMMM;
    int             index_offset = 0;
    int             qm_nr        = 0;
    // Counters of the removed force-field terms, summed over the molecule types
    //   that contain QM atoms, reported at the end of this routine.
    QmmmRemovedInteractions removed;
    // Per-atom record of the same changes, written to a separate file.
    QmmmTopologyReport report;

    // LJ between the QM and the MM atoms: by the exclusion rules of the force field
    //   (default), or, as in earlier versions, with the boundary MM atoms excluded
    //   from every QM atom.
    bool        excludeBoundaryLJ = false;
    const char* ljEnv             = std::getenv("GMX_QMMM_LJ_SCHEME");
    if (ljEnv != nullptr && gmx_strcasecmp(ljEnv, "exclude") == 0)
    {
        excludeBoundaryLJ = true;
    }
    else if (ljEnv != nullptr && gmx_strcasecmp(ljEnv, "forcefield") != 0)
    {
        gmx_fatal(FARGS,
                  "Unknown value '%s' of the environment variable GMX_QMMM_LJ_SCHEME. "
                  "Use 'forcefield' (the default) or 'exclude'.",
                  ljEnv);
    }
    if (qmmmMode == GmxQmmmMode::GMX_QMMM_ORIGINAL)
    {
        GMX_LOG(logger.info)
                .asParagraph()
                .appendTextFormatted(
                        excludeBoundaryLJ
                                ? "QM/MM: GMX_QMMM_LJ_SCHEME=exclude -- the LJ and LJ-14 "
                                  "interactions of every QM atom with the MM atoms bound to the "
                                  "QM region are excluded (the behaviour of earlier versions)."
                                : "QM/MM: the LJ between the QM and the MM atoms follows the "
                                  "exclusion rules of the force field (nrexcl and [ pairs ]); "
                                  "only the LJ within the QM region is excluded. To exclude also "
                                  "the LJ of the MM atoms bound to the QM region with every QM "
                                  "atom, set GMX_QMMM_LJ_SCHEME=exclude.");
    }

    grpnr = sys->groups.groupNumbers[SimulationAtomGroupType::QuantumMechanics].data();

    for (size_t mb = 0; mb < sys->molblock.size(); mb++)
    {
        molb    = &sys->molblock[mb];
        nat_mol = sys->moltype[molb->type].atoms.nr;
        for (mol = 0; mol < molb->nmol; mol++)
        {
            bQMMM = FALSE;
            for (int i = 0; i < nat_mol; i++)
            {
                if ((grpnr ? grpnr[i] : 0) < (ir->opts.ngQM))
                {
                    bQMMM = TRUE;
                    qm_nr++;
                }
            }

            if (bQMMM)
            {
                nr_mol_with_qm_atoms++;
                if (molb->nmol > 1)
                {
                    /* We need to split this molblock */
                    if (mol > 0)
                    {
                        /* Split the molblock at this molecule */
                        auto pos = sys->molblock.begin() + mb + 1;
                        sys->molblock.insert(pos, sys->molblock[mb]);
                        sys->molblock[mb].nmol = mol;
                        sys->molblock[mb + 1].nmol -= mol;
                        mb++;
                        molb = &sys->molblock[mb];
                    }
                    if (molb->nmol > 1)
                    {
                        /* Split the molblock after this molecule */
                        auto pos = sys->molblock.begin() + mb + 1;
                        sys->molblock.insert(pos, sys->molblock[mb]);
                        molb                   = &sys->molblock[mb];
                        sys->molblock[mb].nmol = 1;
                        sys->molblock[mb + 1].nmol -= 1;
                    }

                    /* Create a copy of a moltype for a molecule
                     * containing QM atoms and append it in the end of the list
                     */
                    std::vector<gmx_moltype_t> temp(sys->moltype.size());
                    for (size_t i = 0; i < sys->moltype.size(); ++i)
                    {
                        copy_moltype(&sys->moltype[i], &temp[i]);
                    }
                    sys->moltype.resize(sys->moltype.size() + 1);
                    for (size_t i = 0; i < temp.size(); ++i)
                    {
                        copy_moltype(&temp[i], &sys->moltype[i]);
                    }
                    copy_moltype(&sys->moltype[molb->type], &sys->moltype.back());
                    /* Copy the exclusions to a new array, since this is the only
                     * thing that needs to be modified for QMMM.
                     */
                    sys->moltype.back().excls = sys->moltype[molb->type].excls;
                    /* Set the molecule type for the QMMM molblock */
                    molb->type = sys->moltype.size() - 1;
                }
                generate_qmexcl_moltype(&sys->moltype[molb->type], grpnr, ir, qmmmMode, logger,
                                        &removed, index_offset, &report, excludeBoundaryLJ);
            }
            if (grpnr)
            {
                grpnr += nat_mol;
            }
            index_offset += nat_mol;
        }
    }
    if (nr_mol_with_qm_atoms > 0)
    {
        reportQmmmRemovedInteractions(removed, qmmmMode, logger);
    }
    if (nr_mol_with_qm_atoms > 0 && qmmmReportsEnabled())
    {
        const char* reportFile = std::getenv("GMX_QMMM_TOPOLOGY_REPORT");
        if (reportFile == nullptr)
        {
            reportFile = "qmmm_topology_report.txt";
        }
        writeQmmmTopologyReport(report, qmmmMode, reportFile);
        GMX_LOG(logger.info)
                .appendTextFormatted(
                        "QM/MM: every removed term, pair and exclusion is listed atom by atom in %s\n"
                        "       (file name set with GMX_QMMM_TOPOLOGY_REPORT, switched off with "
                        "GMX_QMMM_REPORTS=off).\n",
                        reportFile);
    }
    if (qmmmMode != GmxQmmmMode::GMX_QMMM_MIMIC && nr_mol_with_qm_atoms > 1)
    {
        /* generate a warning is there are QM atoms in different topologies.
         * In this case, it is not possible at this stage to mutualy exclude
         * the non-bonded interactions via the exclusions (AFAIK). Instead,
         * the user is advised to use the energy group exclusions in the mdp file
         */
        warning_note(wi,
                     "\nThe QM subsystem is divided over multiple topologies. "
                     "The mutual non-bonded interactions cannot be excluded. "
                     "There are two ways to achieve this:\n\n"
                     "1) merge the topologies, such that the atoms of the QM "
                     "subsystem are all present in one single topology file. "
                     "In this case this warning will dissappear\n\n"
                     "2) exclude the non-bonded interactions explicitly via the "
                     "energygrp-excl option in the mdp file. if this is the case "
                     "this warning may be ignored"
                     "\n\n");
    }
}
