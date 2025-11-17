import os
import numpy as np
from ..gdf_emit import gdf_emit
from .. import constants


def read_gdf_emit_file_object(self, file):
    if isinstance(file, (str)):
        return gdf_emit(file)
    elif isinstance(file, (gdf_emit)):
        return file
    else:
        raise Exception("file is not str or gdf object!")


def read_gdf_twiss_files(self, filename=None, gdfbeam=None, reset=True):
    if reset:
        self.reset_dicts()
    if isinstance(filename, (list, tuple)):
        for f in filename:
            self.read_gdf_twiss_files(filename=f, reset=False)
    elif os.path.isfile(filename):
        lattice_name = os.path.basename(filename).split(".")[0]
        if gdfbeam is None and filename is not None:
            gdfbeamdata = read_gdf_emit_file_object(self, filename)

        if hasattr(gdfbeamdata, "avgz"):
            # mjohnson code added 2022-08-11
            nsteps = len(gdfbeamdata.avgz)
            z_sort = np.array(
                [x for _, x in sorted(zip(gdfbeamdata.avgt, gdfbeamdata.avgz))],
                dtype=float,
            )
            order = np.array(
                [x for _, x in sorted(zip(gdfbeamdata.avgt, np.arange(nsteps)))],
                dtype=int,
            )

            offset = 0.0
            for i in range(1, nsteps):
                pos = 0.0 if i == 0 else z_sort[i - 1]
                if z_sort[i] < pos:
                    offset += pos

                gdfbeamdata.avgz[order[i]] = z_sort[i] + offset

            # original code begins
            self.z.val = np.append(self.z.val, gdfbeamdata.avgz)
            self.s.val = np.append(self.s.val, gdfbeamdata.avgz)

        elif hasattr(gdfbeamdata, "position"):
            self.append("z", gdfbeamdata.position)
            self.append("s", gdfbeamdata.position)
        cp = self.E0 * np.sqrt(gdfbeamdata.avgG**2 - 1)
        # self.cp.val = np.append(self.cp.val, cp / constants.elementary_charge)
        ke = np.array(
            (np.sqrt(self.E0**2 + cp**2) - self.E0**2) / constants.elementary_charge
        )
        self.kinetic_energy.val = np.append(self.kinetic_energy.val, ke)
        gamma = 1 + ke / self.E0_eV
        cp = ke / np.sqrt((gamma - 1) / (gamma + 1))
        self.cp.val = np.append(self.cp.val, cp)
        self.mean_cp.val = np.append(self.mean_cp.val, cp)
        self.gamma.val = np.append(self.gamma.val, gamma)
        self.p.val = np.append(self.p.val, cp * self.q_over_c)
        self.enx.val = np.append(self.enx.val, gdfbeamdata.nemixrms)
        self.ex.val = np.append(self.ex.val, gdfbeamdata.nemixrms / gdfbeamdata.avgG)
        self.eny.val = np.append(self.eny.val, gdfbeamdata.nemiyrms)
        self.ey.val = np.append(self.ey.val, gdfbeamdata.nemiyrms / gdfbeamdata.avgG)
        self.enz.val = np.append(self.enz.val, gdfbeamdata.nemizrms)
        self.ez.val = np.append(self.ez.val, gdfbeamdata.nemizrms / gdfbeamdata.avgG)
        self.beta_x.val = np.append(self.beta_x.val, gdfbeamdata.CSbetax)
        self.alpha_x.val = np.append(self.alpha_x.val, gdfbeamdata.CSalphax)
        self.gamma_x.val = np.append(
            self.gamma_x.val, (1 + gdfbeamdata.CSalphax**2) / gdfbeamdata.CSbetax
        )
        self.beta_y.val = np.append(self.beta_y.val, gdfbeamdata.CSbetay)
        self.alpha_y.val = np.append(self.alpha_y.val, gdfbeamdata.CSalphay)
        self.gamma_y.val = np.append(
            self.gamma_y.val, (1 + gdfbeamdata.CSalphay**2) / gdfbeamdata.CSbetay
        )
        self.beta_z.val = np.append(self.beta_z.val, np.zeros(len(gdfbeamdata.stdx)))
        self.alpha_z.val = np.append(self.alpha_z.val, np.zeros(len(gdfbeamdata.stdx)))
        self.gamma_z.val = np.append(self.gamma_z.val, np.zeros(len(gdfbeamdata.stdx)))
        self.sigma_x.val = np.append(self.sigma_x.val, gdfbeamdata.stdx)
        self.sigma_y.val = np.append(self.sigma_y.val, gdfbeamdata.stdy)
        self.sigma_xp.val = np.append(
            self.sigma_xp.val, gdfbeamdata.stdBx / gdfbeamdata.avgBz
        )
        self.sigma_yp.val = np.append(
            self.sigma_yp.val, gdfbeamdata.stdx / gdfbeamdata.avgBz
        )
        self.mean_x.val = np.append(self.mean_x.val, gdfbeamdata.avgx)
        self.mean_y.val = np.append(self.mean_y.val, gdfbeamdata.avgy)
        beta = np.sqrt(1 - (gamma**-2))
        if hasattr(gdfbeamdata, "stdt"):
            self.sigma_t.val = np.append(self.sigma_t.val, gdfbeamdata.stdt)
        else:
            self.sigma_t.val = np.append(
                self.sigma_t.val, gdfbeamdata.stdz / (beta * constants.speed_of_light)
            )
        if hasattr(gdfbeamdata, "avgt"):
            self.t.val = np.append(self.t.val, gdfbeamdata.avgt)
        else:
            self.t.val = np.append(self.t.val, gdfbeamdata.time)
        self.sigma_z.val = np.append(self.sigma_z.val, gdfbeamdata.stdz)
        # self.append('sigma_cp', (gdfbeamdata.stdG / gdfbeamdata.avgG) * cp)
        self.sigma_cp.val = np.append(
            self.sigma_cp.val,
            (gdfbeamdata.stdG / gdfbeamdata.avgG) * cp / constants.elementary_charge,
        )
        self.sigma_p.val = np.append(
            self.sigma_p.val, (gdfbeamdata.stdG / gdfbeamdata.avgG)
        )
        self.mux.val = np.append(self.mux.val, np.zeros(len(gdfbeamdata.stdx)))
        self.muy.val = np.append(self.muy.val, np.zeros(len(gdfbeamdata.stdx)))
        self.eta_x.val = np.append(self.eta_x.val, np.zeros(len(gdfbeamdata.stdx)))
        self.eta_xp.val = np.append(self.eta_xp.val, np.zeros(len(gdfbeamdata.stdx)))
        self.eta_y.val = np.append(self.eta_y.val, np.zeros(len(gdfbeamdata.stdy)))
        self.eta_yp.val = np.append(self.eta_yp.val, np.zeros(len(gdfbeamdata.stdy)))
        self.element_name.val = np.append(
            self.element_name.val, np.full(len(gdfbeamdata.stdx), "")
        )
        self.lattice_name.val = np.append(
            self.lattice_name.val, np.full(len(gdfbeamdata.stdx), lattice_name)
        )
        # ## BEAM parameters
        self.ecnx.val = np.append(self.ecnx.val, np.zeros(len(gdfbeamdata.stdx)))
        self.ecny.val = np.append(self.ecny.val, np.zeros(len(gdfbeamdata.stdx)))
        self.eta_x_beam.val = np.append(
            self.eta_x_beam.val, np.zeros(len(gdfbeamdata.stdx))
        )
        self.eta_xp_beam.val = np.append(
            self.eta_xp_beam.val, np.zeros(len(gdfbeamdata.stdx))
        )
        self.eta_y_beam.val = np.append(
            self.eta_y_beam.val, np.zeros(len(gdfbeamdata.stdx))
        )
        self.eta_yp_beam.val = np.append(
            self.eta_yp_beam.val, np.zeros(len(gdfbeamdata.stdx))
        )
        self.beta_x_beam.val = np.append(
            self.beta_x_beam.val, np.zeros(len(gdfbeamdata.stdx))
        )
        self.beta_y_beam.val = np.append(
            self.beta_y_beam.val, np.zeros(len(gdfbeamdata.stdx))
        )
        self.alpha_x_beam.val = np.append(
            self.alpha_x_beam.val, np.zeros(len(gdfbeamdata.stdx))
        )
        self.alpha_y_beam.val = np.append(
            self.alpha_y_beam.val, np.zeros(len(gdfbeamdata.stdx))
        )
