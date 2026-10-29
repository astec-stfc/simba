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
        gdfbeamdata = read_gdf_emit_file_object(
            self, filename if gdfbeam is None else gdfbeam
        )

        if hasattr(gdfbeamdata, "avgz"):
            # avgz resets at each new lattice section; unwrap it in time order
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

            self.append_columns(nsteps, z=gdfbeamdata.avgz, s=gdfbeamdata.avgz)

        elif hasattr(gdfbeamdata, "position"):
            self.append("z", gdfbeamdata.position)
            self.append("s", gdfbeamdata.position)
        cp = self.E0 * np.sqrt(gdfbeamdata.avgG**2 - 1)
        ke = np.array(
            (np.sqrt(self.E0**2 + cp**2) - self.E0) / constants.elementary_charge
        )
        gamma = 1 + ke / self.E0_eV
        cp = ke / np.sqrt((gamma - 1) / (gamma + 1))
        beta = np.sqrt(1 - (gamma**-2))
        if hasattr(gdfbeamdata, "stdt"):
            sigma_t = gdfbeamdata.stdt
        else:
            sigma_t = gdfbeamdata.stdz / (beta * constants.speed_of_light)
        self.append_columns(
            len(gdfbeamdata.stdx),
            kinetic_energy=ke,
            cp=cp,
            mean_cp=cp,
            gamma=gamma,
            p=cp * self.q_over_c,
            enx=gdfbeamdata.nemixrms,
            ex=gdfbeamdata.nemixrms / gdfbeamdata.avgG,
            eny=gdfbeamdata.nemiyrms,
            ey=gdfbeamdata.nemiyrms / gdfbeamdata.avgG,
            enz=gdfbeamdata.nemizrms,
            ez=gdfbeamdata.nemizrms / gdfbeamdata.avgG,
            beta_x=gdfbeamdata.CSbetax,
            alpha_x=gdfbeamdata.CSalphax,
            gamma_x=(1 + gdfbeamdata.CSalphax**2) / gdfbeamdata.CSbetax,
            beta_y=gdfbeamdata.CSbetay,
            alpha_y=gdfbeamdata.CSalphay,
            gamma_y=(1 + gdfbeamdata.CSalphay**2) / gdfbeamdata.CSbetay,
            beta_z=0.0, alpha_z=0.0, gamma_z=0.0,
            sigma_x=gdfbeamdata.stdx,
            sigma_y=gdfbeamdata.stdy,
            sigma_xp=gdfbeamdata.stdBx / gdfbeamdata.avgBz,
            sigma_yp=gdfbeamdata.stdx / gdfbeamdata.avgBz,
            mean_x=gdfbeamdata.avgx,
            mean_y=gdfbeamdata.avgy,
            sigma_t=sigma_t,
            t=gdfbeamdata.avgt if hasattr(gdfbeamdata, "avgt") else gdfbeamdata.time,
            sigma_z=gdfbeamdata.stdz,
            sigma_cp=(gdfbeamdata.stdG / gdfbeamdata.avgG) * cp / constants.elementary_charge,
            sigma_p=(gdfbeamdata.stdG / gdfbeamdata.avgG),
            mux=0.0, muy=0.0, eta_x=0.0, eta_xp=0.0, eta_y=0.0, eta_yp=0.0,
            element_name="",
            lattice_name=lattice_name,
            # ## BEAM parameters
            ecnx=0.0, ecny=0.0,
            eta_x_beam=0.0, eta_xp_beam=0.0, eta_y_beam=0.0, eta_yp_beam=0.0,
            beta_x_beam=0.0, beta_y_beam=0.0, alpha_x_beam=0.0, alpha_y_beam=0.0,
        )
