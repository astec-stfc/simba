"""Microbunching-gain (MBI) physics process for Ocelot."""

import numpy as np
from ocelot.common.globals import I_Alfven, Z0, m_e_MeV, q_e, speed_of_light
from ocelot.cpbd.elements import Bend, RBend, SBend
from ocelot.cpbd.beam import ParticleArray
from scipy.special import kv, iv
from ocelot.cpbd.physics_proc import PhysProc
from ocelot.cpbd.beam import global_slice_analysis
from ocelot.cpbd.magnetic_lattice import MagneticLattice
from ocelot.cpbd.elements.element import Element
from ocelot.cpbd.tm_utils import transfer_maps_mult
import logging
import copy

logger = logging.getLogger(__name__)

A_csr = -0.94 + 1.63j


def lattice_transfer_map_z(lattice, energy, zfin):
    """
    First-order transfer map of `lattice` from its start to `zfin`.

    :param lattice: MagneticLattice
    :param energy: initial beam energy [GeV]
    :param zfin: final position [m]
    :return: the 6x6 matrix, and the element at `zfin`
    """

    Ra = np.eye(6)
    Ta = np.zeros((6, 6, 6))
    Ba = np.zeros((6, 1))
    E = energy
    i = 0
    elem = lattice.sequence[i]
    L = elem.l
    while zfin > L:
        for Rb, Bb, Tb, tm in zip(elem.R(E), elem.B(E), elem.T(E), elem.tms):
            Ba, Ra, Ta = transfer_maps_mult(Ba, Ra, Ta, Bb, Rb, Tb)
            E += tm.get_delta_e()
        i += 1
        elem = lattice.sequence[i]
        L += elem.l
    delta_l = zfin - (L - elem.l)
    first_order_tms = np.dot(
        lattice.sequence[i]
        .get_section_tms(start_l=0.0, delta_l=delta_l, first_order_only=True)[-1]
        .get_params(E)
        .R,
        Ra,
    )

    return first_order_tms, elem


def k_wn(lamb: float):
    """
    Wave number [1/m] of wavelength `lamb` [m].
    """
    return 2 * np.pi / lamb


def b0(lamb: float, I0: float):
    """
    Initial longitudinal bunching factor from shot noise.

    :param lamb: initial modulation wavelength [m]
    :param I0: beam current [A]
    """
    return np.sqrt(q_e * speed_of_light / (I0 * lamb))


def p0(lamb: float, compression_factor: float, sdelta: float, r56: float, b0: float):
    """
    Initial momentum bunching factor from shot noise.

    :param lamb: initial modulation wavelength [m]
    :param compression_factor: longitudinal compression factor
    :param sdelta: fractional energy spread
    :param r56: R56 from the start to the current step [m]
    :param b0: bunching factor from shot noise (see :func:`b0`)
    """
    kfac = k_wn(lamb / compression_factor)
    r56fac = r56
    sigdfac = sdelta**2
    return -1j * kfac * r56fac * sigdfac * b0


class MBI(PhysProc):
    """
    Microbunching-gain physics process, following `Tsai et al`_ (equation numbers refer to it).

    Each step sums the integral kernel over all previous steps into the bunching factors,
    which are set on the ``ParticleArray`` as ``bf`` and ``pf``.

    .. _Tsai et al: https://journals.aps.org/prab/abstract/10.1103/PhysRevAccelBeams.23.124401

    Parameters
    ----------
    lattice: MagneticLattice
        Lattice used during tracking
    step: float
        Step of the calculation [Navigator.unit_step]
    lamb_range: list
        Initial modulation wavelengths [m]
    slices: int
        Number of time slices; zero takes only the central slice
    lsc: bool
        Include LSC impedance
    csr: bool
        Include CSR impedance
    """

    def __init__(
        self,
        lattice: MagneticLattice,
        step: float = 1,
        lamb_range: list = (1e-6, 50e-6, 10),
        slices: int = 0,
        lsc: bool = True,
        csr: bool = True,
    ):
        PhysProc.__init__(self, step)
        self.lattice = lattice
        self.lsc = lsc
        self.csr = csr
        self.first = True
        self.lamb_range = lamb_range
        self.slice = int(slices)
        self.dist = []
        self.bf = [[] for i in range(len(self.lamb_range))]
        self.pf = [[] for i in range(len(self.lamb_range))]
        self.slice_params = []
        if self.slice > 0:
            self.bf = [
                [[] for i in range(len(self.lamb_range))] for j in range(self.slice)
            ]
            self.pf = [
                [[] for i in range(len(self.lamb_range))] for j in range(self.slice)
            ]
            self.slice_params = [[] for i in range(self.slice)]
        self.optics_map = []

    def set_lamb_range(self, lamb_range) -> None:
        """
        Set the initial modulation wavelengths [m] and reset the bunching factors.
        """
        self.lamb_range = lamb_range
        self.bf = [[] for i in range(len(self.lamb_range))]
        self.pf = [[] for i in range(len(self.lamb_range))]

    def set_slice(self, slices) -> None:
        """
        Set the number of slices along the beam (zero for the core only).
        """
        if slices > 0:
            self.slice = slices
            self.bf = [
                [[] for i in range(len(self.lamb_range))] for j in range(self.slice)
            ]
            self.pf = [
                [[] for i in range(len(self.lamb_range))] for j in range(self.slice)
            ]
            self.slice_params = [[] for i in range(self.slice)]
        else:
            self.bf = [[] for i in range(len(self.lamb_range))]
            self.pf = [[] for i in range(len(self.lamb_range))]
            self.slice_params = []

    def get_slice_params(self, p_array: ParticleArray, slices: bool = False) -> dict:
        """
        Mean slice parameters from Ocelot's ``global_slice_analysis``.

        :param p_array: the beam
        :param slices: average over all slices if True, else over the central ones
        :return: dict of slice parameters
        """
        slice_params = global_slice_analysis(p_array)
        if not slices:
            sli_cen = len(slice_params.s) / 2
            sli_tot = len(slice_params.s)
            sli_min = int(sli_cen - (sli_tot / 2.5))
            sli_max = int(sli_cen + (sli_tot / 2.5))
        else:
            sli_min = 0
            sli_max = -1
        sli = {}
        for k, v in slice_params.__dict__.items():
            if k in ["beta_x", "alpha_x", "beta_y", "alpha_y", "I", "sig_x", "sig_y"]:
                sli.update({k: np.mean(v[sli_min:sli_max])})
            elif k == "se":
                sli.update(
                    {
                        "sdelta": np.mean(
                            v[sli_min:sli_max] / slice_params.me[sli_min:sli_max]
                        )
                    }
                )
            elif k == "me":
                sli.update({"me": np.mean(v[sli_min:sli_max] / 1e9)})
                sli.update({"gamma": np.mean(v[sli_min:sli_max] / 1e6 / m_e_MeV)})
            elif k in ["ex", "ey"]:
                sli.update(
                    {
                        k: np.mean(v[sli_min:sli_max])
                        * 1e-6
                        / (np.mean(slice_params.me[sli_min:sli_max] / 1e6 / m_e_MeV))
                    }
                )
        sli.update({"s": p_array.s})
        return sli

    def apply(self, p_array: ParticleArray, dz: float) -> None:
        """
        Record slice parameters and the optics map, then update ``bf`` and ``pf``.

        :param p_array: the beam
        :param dz: step [m]
        """
        if dz < 1e-10:
            logger.debug(" MBI applied, dz < 1e-10, dz = " + str(dz))
            return
        logger.debug(" MBI applied, dz =" + str(dz))
        p_array_c = copy.deepcopy(p_array)
        if self.slice > 0:
            t_his = np.histogram(p_array_c.tau(), bins=self.slice)[1]
            ltm = []
            for i in range(len(t_his) - 1):
                inds = np.where(
                    np.logical_and(
                        p_array_c.tau() >= t_his[i], p_array_c.tau() <= t_his[i + 1]
                    )
                )[0]
                pn = copy.deepcopy(p_array_c)
                pn.rparticles = p_array_c.rparticles[:, inds]
                pn.q_array = p_array_c.q_array[inds]
                self.slice_params[i].append(self.get_slice_params(pn, slices=True))
                z0 = self.slice_params[i][-1]["s"] - self.slice_params[i][0]["s"]
                res = lattice_transfer_map_z(
                    self.lattice, self.slice_params[i][0]["me"], z0
                )
                ltm.append(res[0])
                elem = res[1]
            self.optics_map.append(ltm[-1])
        else:
            self.slice_params.append(self.get_slice_params(p_array_c))
            z0 = self.slice_params[-1]["s"] - self.slice_params[0]["s"]
            ltm, elem = lattice_transfer_map_z(
                self.lattice, self.slice_params[0]["me"], z0
            )
            self.optics_map.append(ltm)
        self.dist.append(self.z0)
        if self.slice > 0:
            p_array.bf, p_array.pf = self.get_bf_slice(
                self.lamb_range, self.slice_params, self.optics_map, self.slice, elem
            )
        else:
            p_array.bf, p_array.pf = self.get_bf(
                self.lamb_range, self.slice_params, self.optics_map, elem
            )

    def get_bf(
        self, lamb_range: list, slice_params: list, optics_map: list, elem: Element
    ) -> list:
        """
        Bunching factors at the current position (Eq. (58)).

        :param lamb_range: initial modulation wavelengths [m]
        :param slice_params: slice-parameter dicts, one per step
        :param optics_map: first-order transfer matrices, one per step
        :param elem: Ocelot element at the current position
        :return: bunching factors in z and p
        """
        cur_init = slice_params[0]["I"]
        for i, l in enumerate(lamb_range):
            cur_now = slice_params[-1]["I"]
            sdelta = slice_params[-1]["sdelta"]
            r56 = optics_map[-1][4, 5]
            ld_0s = self.ld0s(l, slice_params, optics_map)
            b0fac = b0(l, cur_init) * ld_0s
            p0fac = abs(p0(l, cur_now / cur_init, sdelta, r56, b0fac))
            self.bf[i].append(b0fac)
            self.pf[i].append(p0fac)
            if len(self.slice_params) > 1:
                k0tot = []
                k1tot = []
                k2tot = []
                for j, b in enumerate(self.bf[i]):
                    fac = 0.5 if j == 0 else 1
                    distance = (
                        self.slice_params[-1]["s"]
                        if j == 0
                        else (slice_params[j]["s"] - slice_params[j - 1]["s"])
                    )
                    k0, k1, k2 = self.kernels(l, slice_params, optics_map, j, elem)
                    k0r = distance * fac * k0
                    k0tot.append(abs((k0r.real * b) + (k0r.imag * b)))
                    k1r = distance * fac * k1
                    k1tot.append(abs((k1r.real * b) + (k1r.imag * b)))
                    k2r = distance * fac * k2
                    k2tot.append(abs((k2r.real * b) + (k2r.imag * b)))
                    k2tot[-1] *= slice_params[j]["sdelta"] ** 2
                    k0tot[-1] *= self.bf[i][j]
                    k2tot[-1] *= self.bf[i][j]
                self.bf[i][-1] += np.nansum(k1tot)
                self.pf[i][-1] -= abs(np.nansum(k0tot)) - abs(np.nansum(k2tot))
        return [self.bf, self.pf]

    def get_bf_slice(
        self,
        lamb_range: list,
        slice_params: list,
        optics_map: list,
        slices: int,
        elem: Element,
    ) -> list:
        """
        Bunching factors at the current position (Eq. (58)), per longitudinal slice.

        :param lamb_range: initial modulation wavelengths [m]
        :param slice_params: per slice, slice-parameter dicts, one per step
        :param optics_map: first-order transfer matrices, one per step
        :param slices: number of slices
        :param elem: Ocelot element at the current position
        :return: bunching factors in z and p
        """
        for h in range(slices):
            cur_init = slice_params[h][0]["I"]
            sli = list(slice_params[h])
            for i, l in enumerate(lamb_range):
                ld_0s = self.ld0s(l, sli, optics_map)
                cur_now = sli[-1]["I"]
                sdelta = sli[-1]["sdelta"]
                r56 = optics_map[-1][4, 5]
                b0fac = b0(l, sli[0]["I"]) * ld_0s
                p0fac = abs(p0(l, cur_now / cur_init, sdelta, r56, b0fac))
                self.bf[h][i].append(b0fac)
                self.pf[h][i].append(p0fac)
                if np.array(slice_params).shape[1] > 1:
                    k0tot = []
                    k1tot = []
                    k2tot = []
                    for j, b in enumerate(self.bf[h][i]):
                        fac = 0.5 if j == 0 else 1
                        distance = (
                            sli[-1]["s"] if j == 0 else (sli[j]["s"] - sli[j - 1]["s"])
                        )
                        k0, k1, k2 = self.kernels(l, sli, optics_map, j, elem)
                        k0r = distance * fac * k0
                        k0tot.append(abs((k0r.real * b) + (k0r.imag * b)))
                        k1r = distance * fac * k1
                        k1tot.append(abs((k1r.real * b) + (k1r.imag * b)))
                        k2r = distance * fac * k2
                        k2tot.append(abs((k2r.real * b) + (k2r.imag * b)))
                    self.bf[h][i][-1] += np.nansum(k1tot)
        return [self.bf, self.pf]

    def ld0s(self, lamb: list, slice_params: list, optics_map: list) -> float:
        """
        Landau damping LD(0->s) from the start to the current step (Eq. (46)).

        :param lamb: initial modulation wavelength [m]
        :param slice_params: slice-parameter dicts, one per step
        :param optics_map: first-order transfer matrices, one per step
        """
        compfac = slice_params[-1]["I"] / slice_params[0]["I"]
        kfac = -(k_wn(lamb * compfac) ** 2) / 2
        ex = slice_params[0]["ex"]
        ey = slice_params[0]["ey"]
        betax = slice_params[0]["beta_x"]
        betay = slice_params[0]["beta_y"]
        alphax = slice_params[0]["alpha_x"]
        alphay = slice_params[0]["alpha_y"]
        sigd = slice_params[0]["sdelta"] * 2
        r51s = optics_map[-1][4, 0]
        r52s = optics_map[-1][4, 1]
        r53s = optics_map[-1][4, 2]
        r54s = optics_map[-1][4, 3]
        r56s = optics_map[-1][4, 5]
        r51r52 = (r51s - ((alphax / betax) * r52s)) ** 2
        r52 = r52s**2
        r53r54 = (r53s - ((alphay / betay) * r54s)) ** 2
        r54 = r54s**2
        r56 = r56s**2
        exbeta = ex * betax
        eybeta = ey * betay
        exobeta = ex / betax
        eyobeta = ey / betay
        exponent = (
            (exbeta * r51r52)
            + (exobeta * r52)
            + (eybeta * r53r54)
            + (eyobeta * r54)
            + ((sigd**2) * r56)
        )
        return np.exp(kfac * exponent)

    def ldtaus(
        self, lamb: list, slice_params: list, optics_map: list, i1: int
    ) -> float:
        """
        Landau damping LD(tau->s) from step `i1` to the current step (Eq. (50)).

        :param lamb: initial modulation wavelength [m]
        :param slice_params: slice-parameter dicts, one per step
        :param optics_map: first-order transfer matrices, one per step
        :param i1: index of step tau
        """
        kfac = -(k_wn(lamb) ** 2) / 2
        ex = slice_params[0]["ex"]
        ey = slice_params[0]["ey"]
        betax = slice_params[0]["beta_x"]
        betay = slice_params[0]["beta_y"]
        alphax = slice_params[0]["alpha_x"]
        alphay = slice_params[0]["alpha_y"]
        sigd = slice_params[i1]["sdelta"] * 2
        r51s = optics_map[-1][4, 0]
        r52s = optics_map[-1][4, 1]
        r53s = optics_map[-1][4, 2]
        r54s = optics_map[-1][4, 3]
        r56s = optics_map[-1][4, 5]
        r51tau = optics_map[i1][4, 0]
        r52tau = optics_map[i1][4, 1]
        r53tau = optics_map[i1][4, 2]
        r54tau = optics_map[i1][4, 3]
        r56tau = optics_map[i1][4, 5]
        r51taus = r51s - r51tau
        r52taus = r52s - r52tau
        r53taus = r53s - r53tau
        r54taus = r54s - r54tau
        r56taus = r56s - r56tau
        r51r52 = (r51taus - ((alphax / betax) * r52taus)) ** 2
        r52 = r52taus**2
        r53r54 = (r53taus - ((alphay / betay) * r54taus)) ** 2
        r54 = r54taus**2
        r56 = r56taus**2
        exbeta = ex * betax
        eybeta = ey * betay
        exobeta = ex / betax
        eyobeta = ey / betay
        exponent = (
            (exbeta * r51r52)
            + (exobeta * r52)
            + (eybeta * r53r54)
            + (eyobeta * r54)
            + ((sigd**2) * r56)
        )
        return np.exp(kfac * exponent)

    def r56taus(self, optics_map: list, i1: int) -> float:
        """
        R56(tau->s) from step `i1` to the current step (Eq. (49)).

        :param optics_map: first-order transfer matrices, one per step
        :param i1: index of step tau
        """
        r51s = optics_map[i1][4, 0]
        r52s = optics_map[i1][4, 1]
        r53s = optics_map[i1][4, 2]
        r54s = optics_map[i1][4, 3]
        r55s = optics_map[i1][4, 4]
        r56s = optics_map[i1][4, 5]
        r51tau = optics_map[-1][4, 0]
        r52tau = optics_map[-1][4, 1]
        r53tau = optics_map[-1][4, 2]
        r54tau = optics_map[-1][4, 3]
        r55tau = optics_map[-1][4, 4]
        r56tau = optics_map[-1][4, 5]
        return (
            (r56s * r55tau)
            - (r56tau * r55s)
            + (r51tau * r52s)
            - (r51s * r52tau)
            + (r53tau * r54s)
            - (r53s * r54tau)
        )

    def kernels(
        self, lamb: float, slice_params: list, optics_map: list, i1: int, elem: Element
    ) -> list:
        """
        Kernels K0, K1, K2 (Eqs. (A17a-c)).

        :param lamb: initial modulation wavelength [m]
        :param slice_params: slice-parameter dicts, one per step
        :param optics_map: first-order transfer matrices, one per step
        :param i1: index of step tau
        :param elem: Ocelot element at the current position
        """
        currentfac = slice_params[i1]["I"] / ((slice_params[i1]["gamma"]) * I_Alfven)
        compfac = slice_params[i1]["I"] / slice_params[0]["I"]
        lamb_compressed = lamb / compfac
        impedancefac = (
            self.lscimpedance(lamb_compressed, slice_params, i1) if self.lsc else 0
        )
        if self.csr and (elem.__class__ in [RBend, SBend, Bend]):
            impedancefac += self.csrimpedance(lamb_compressed, elem)
        ldfac = self.ldtaus(lamb_compressed, slice_params, optics_map, i1)
        k0 = currentfac * impedancefac * ldfac
        kfac = k_wn(lamb_compressed)
        r56fac = self.r56taus(optics_map, i1)
        k1 = k0 * kfac * r56fac
        k2 = k0 * (kfac * r56fac) ** 2
        return [k0, k1, k2]

    def lscimpedance(self, lamb: float, slice_params: list, i1: int) -> float:
        """
        LSC impedance; in place of Eq. (52), PRAB 23, 014403 Eq. (26).

        :param lamb: modulation wavelength [m]
        :param slice_params: slice-parameter dicts, one per step
        :param i1: index of the step
        """
        kz = k_wn(lamb)
        rb = 0.8375 * (slice_params[i1]["sig_x"] + slice_params[i1]["sig_y"])
        gamma = slice_params[i1]["gamma"]
        xib = kz * rb / gamma
        return (
            1j
            * (Z0 / (np.pi * gamma * rb))
            * (1 - (xib * kv(1, xib) * iv(0, xib)))
            / xib
        )

    def csrimpedance(self, lamb: float, elem: Element) -> float:
        """
        CSR impedance of `elem` (Eq. (51)); zero if it does not bend.

        :param lamb: modulation wavelength [m]
        :param elem: Ocelot element
        """
        if not hasattr(elem, "angle"):
            return 0
        else:
            if abs(elem.angle) < 1e-10:
                return 0
            else:
                kz = k_wn(lamb)
                bendradius = elem.l / abs(elem.angle)
                return -1j * A_csr * (kz ** (1 / 3)) / (bendradius ** (2 / 3))
