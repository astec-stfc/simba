from ...Framework_objects import frameworkLattice
import requests
from confluent_kafka import Consumer, Message
import time, uuid
from pydantic import BaseModel
import numpy as np
from copy import deepcopy
from typing import Dict
from laura.models.element import Diagnostic
from ...Codes.Generators.Generators import frameworkGenerator
from ...Modules.constants import speed_of_light

transform_ocelot_twiss = {
    '_emit_xn': "ecnx",
    '_emit_yn': "ecny",
    '_E': ["cp", 1e-9],
    '_beta_x': "beta_x",
    '_beta_y': "beta_y",
    '_alpha_x': "alpha_x",
    '_alpha_y': "alpha_y",
    'Dx': "eta_x",
    'Dy': "eta_y",
    'Dxp': "eta_xp",
    'Dyp': "eta_yp",
    'mux': "mux",
    'muy': "muy",
    's': "s",
    'x': "mean_x",
    'y': "mean_y",
    'xx': "sigma_x",
    'pxpx': "sigma_xp",
    'yy': "sigma_y",
    'pypy': "sigma_yp",
}


class polylithicLattice(frameworkLattice):
    """
    Class for defining the PolyLithic lattice object, used for running models through polylithic & kafka
    Primarily used for Machine Learning models, but will do anything polylithic can do.
    Required setting in the YAML: "model" to a registered name on polylithic server
    Required setting in executables.yaml: "polylithic_url"
    """

    code: str = "polylithic"
    """String indicating the lattice object type"""

    model: str = None
    """String which is the model name on the PolyLithic server"""

    model_output: Dict = {}
    """Dict of model output data returned from the model"""

    output_schema: Dict = {}
    """Output schema for the model returned from poly-lithic used to determine how to interpret the model output"""

    class ModelParams(BaseModel):
        """
        Model parameters from the polylithic fastapi
        """
        model: str
        """Lattice section in which to execute the model"""

        beam_properties: dict | list
        """Dict of beam and machine properties passed to the model"""

        machine_settings: dict | list
        """Dict of predicted outputs provided by the model"""


    def model_post_init(self, __context):
        super().model_post_init(__context)
        if "model" in self.file_block:
            self.model = self.file_block["model"]
        else:
            raise ValueError(f"\"model\" not provided for {self.objectname}")
        if (
                "input" in self.file_block
                and "particle_definition" in self.file_block["input"]
        ):
            self.particle_definition = "laser"

    def writeElements(self):
        """Not implemented for this class"""
        pass

    def write(self):
        """Not implemented for this class"""
        pass

    def preProcess(self) -> None:
        """
        Prepare the input distribution for ELEGANT based on the `prefix` in the settings
        file for this lattice section, and create the ELEGANT command files.
        """
        super().preProcess()
        prefix = self.get_prefix()
        self.read_input_file(prefix, self.particle_definition)
        self.ref_idx = self.global_parameters["beam"].reference_particle_index
        self.global_parameters["beam"].beam.rematchXPlane(
            **self.initial_twiss["horizontal"]
        )
        self.global_parameters["beam"].beam.rematchYPlane(
            **self.initial_twiss["vertical"]
        )

    def postProcess(self):
        """
        Save the beam file(s) from the polylithic output into HDF5 format
        """
        super().postProcess()
        beamflag = True
        # if "beam" in self.model_output:
        #     self.reconstruct_beams()
        #     beamflag = True
        # else:
        #     beamflag = False
        # for key in self.model_output:
            # if key == "twiss":
        self.reconstruct_twiss(beam=beamflag)

    def run(self):
        """
        Run the polylithic query, schedule the kafka job and wait for it to complete
        Will throw errors if the model doesn't support the specified lattice settings
        """
        debug_no_remote_server = False  # when testing, set this to True to skip the remote server query

        # error handling: if kafka chokes, raise a hard error and crash out
        # query polylithic fastapi for desired beam format using requests
        url = f"{self.executables.polylithic_url}/get_settings/{self.model}"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            json_string = response.json()
            model_requirements = self.ModelParams.model_validate(json_string)
            # if self.objectname != model_requirements.lattice_section:
            #     raise ValueError("Specified model does not match the lattice section, did you use the wrong model?")
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to query polylithic FastAPI: {e}")

        # format beam for requested model data
        beamdata = {}

        beamcopy = deepcopy(self.global_parameters["beam"])
        for key in model_requirements.beam_properties:
            splitkey = key.split(':')
            if len(splitkey) == 2:
                val = getattr(getattr(beamcopy, splitkey[0]), splitkey[1]).val
                if val.ndim == 0:
                    val = float(val)
                beamdata.update({key: val})
            elif len(splitkey) == 3:
                if splitkey[0] not in beamdata:
                    beamdata.update({splitkey[0]: {}})
                if splitkey[1] not in beamdata[splitkey[0]]:
                    beamdata[splitkey[0]].update({splitkey[1]: {}})
                beamattr = getattr(beamcopy, splitkey[0])
                arg = [getattr(beamattr, a) for a in splitkey[2].split(',')]
                val = getattr(beamattr, splitkey[1])(arg[0], arg[1]).val
                if val.ndim == 0:
                    val = float(val)
                beamdata[splitkey[0]][splitkey[1]].update({splitkey[2].split(',')[0]: {splitkey[2].split(',')[1]: val}})

        machinedata = {}
        for key in model_requirements.machine_settings:
            splitkey = key.split(':')
            if len(splitkey) != 2:
                raise ValueError(f"machine_settings must be indexed by elem_name:parameter, not {key}")
            if splitkey[0] not in self.elements:
                raise KeyError(f"Element {splitkey[0]} not found in lattice {self.name}")
            machinedata.update({key: getattr(self.elementObjects[splitkey[0]], splitkey[1])})

        # send kafka the model name, the formatted beam, lattice, and lattice settings
        url = f"{self.executables.polylithic_url}/v2/submit_job/{self.model}"
        job_payload = {
            "model": self.model,
            "job_id": self.model + uuid.uuid4().hex,
            "beam": beamdata,
            "lattice_name": "string",
            "lattice": machinedata,
        }
        try:
            response = requests.post(url, json=job_payload, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to schedule job on polylithic FastAPI: {e}")

        consumer: Consumer = Consumer({'bootstrap.servers': self.executables.kafka_url, 'group.id': "simframe"})
        consumer.subscribe(['pl_job_results'])
        # wait for consumer to return
        start = time.time()
        # while True:
        #     msg: Message = consumer.poll(1.0)
        #     print("poll")
        #     if time.time() - start > 60:
        #         raise TimeoutError("Timeout while waiting for kafka job")
        #     if msg is None:
        #         continue
        #     if msg.error():
        #         raise ValueError(f"Polylithic kafka consumer error: {msg.error()}")
        #
        #     msgvalue = msg.value().decode('utf-8')
        #     print('Received message: {}'.format(msgvalue))
        #     # convert msgvalue to json
        #
        #     msgvalue = json.loads(msgvalue)
        #     if msgvalue["job_id"] == job_payload["job_id"]:
        #         print("job complete")
        #         break
        # consumer.close()

        url = f"{self.executables.polylithic_url}/v2/get_result/{job_payload['job_id']}"
        status = False
        while True:
            if time.time() - start > 60:
                raise TimeoutError("Timeout while waiting to retrieve job results")
            if status:
                break
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                self.model_output = response.json()
                if "job_id" in self.model_output:
                    status = True
            except requests.RequestException as e:
                raise RuntimeError(f"Failed to query result from polylithic FastAPI: {e}")


    def reconstruct_beams(self):
        raise NotImplementedError(
            "Beam reconstruction not yet implemented for polylithic models, only twiss. "
            "Will be added in a future update."
        )

    def reconstruct_twiss(self, beam=False):
        self.model_output.update(
            {
                "twiss":
                    {
                        k: v for k, v in self.model_output["beam"].items() if
                        k in transform_ocelot_twiss.values()
                        or k in [x[0] for x in transform_ocelot_twiss.values()]
                    }
            }
        )
        from ocelot.cpbd.beam import Twiss
        t = Twiss()
        if "s" not in self.model_output["twiss"]:
            anykey = len(self.model_output["twiss"][list(self.model_output["twiss"].keys())[0]])
            self.model_output["twiss"]["s"] = np.linspace(
                self.startObject.physical.start.z,
                self.endObject.physical.end.z,
                anykey
            )
        else:
            if self.model_output["twiss"]["s"][0] < self.startObject.physical.start.z:
                self.model_output["twiss"]["s"] += self.startObject.physical.start.z
        twslen = len(self.model_output["twiss"]["s"])
        twsdat = {e: [] for e in t.__dict__.keys()}
        for k in t.__dict__.keys():
            if k in transform_ocelot_twiss.keys():
                k1 = transform_ocelot_twiss[k][0] if isinstance(transform_ocelot_twiss[k], list) else \
                transform_ocelot_twiss[k]
                if k1 in self.model_output["twiss"]:
                    if isinstance(transform_ocelot_twiss[k], list):
                        kl = transform_ocelot_twiss[k]
                        twsdat[k] = [x * kl[1] for x in self.model_output["twiss"][kl[0]]]
                    else:
                        twsdat[k] = self.model_output["twiss"][k1]
                else:
                    twsdat[k] = np.zeros(twslen)
            else:
                twsdat[k] = np.zeros(twslen)
        np.savez_compressed(
            f'{self.global_parameters["master_subdir"]}/{self.name}_twiss.npz',
            **twsdat,
        )

        if beam:
            init = deepcopy(self.global_parameters["beam"])
            for e in self.elementObjects.values():
                if isinstance(e, Diagnostic):
                    idx = self.find_nearest_idx(self.model_output["beam"]["s"], e.physical.middle.z)
                    gen = frameworkGenerator(
                        global_parameters=self.global_parameters,
                        code="simba",
                        species=init.species,
                        sigma_x=self.model_output["beam"]["sigma_x"][idx] if "sigma_x" in self.model_output[
                            "beam"] else init.sigma_x.val,
                        sigma_y=self.model_output["beam"]["sigma_y"][idx] if "sigma_y" in self.model_output[
                            "beam"] else init.sigma_y.val,
                        sigma_z=self.model_output["beam"]["sigma_z"][idx] if "sigma_z" in self.model_output[
                            "beam"] else init.sigma_z.val,
                        sigma_px=self.model_output["beam"]["sigma_px"][idx] if "sigma_px" in self.model_output[
                            "beam"] else init.sigma_px.val,
                        sigma_py=self.model_output["beam"]["sigma_py"][idx] if "sigma_py" in self.model_output[
                            "beam"] else init.sigma_py.val,
                        sigma_pz=self.model_output["beam"]["sigma_pz"][idx] if "sigma_pz" in self.model_output[
                            "beam"] else init.sigma_pz.val,
                        offset_x=self.model_output["beam"]["offset_x"][idx] if "offset_x" in self.model_output[
                            "beam"] else init.mean_x.val,
                        offset_y=self.model_output["beam"]["offset_y"][idx] if "offset_y" in self.model_output[
                            "beam"] else init.mean_y.val,
                        offset_z=e.physical.middle.z,
                        offset_t=init.mean_t + (e.physical.middle.z - init.mean_z) / speed_of_light,
                        reference_time=init.mean_t + (e.physical.middle.z - init.mean_z) / speed_of_light,
                        number_of_particles=len(init.x),
                        filename=e.name + ".openpmd.hdf5",
                        charge=init.total_charge.val,
                        initial_momentum=self.model_output["beam"]["cp"][idx] if "cp" in self.model_output[
                            "beam"] else init.mean_cp.val,
                    )
                    bea1 = gen.generate()
                    for plane in ["x", "y"]:
                        if f"beta_{plane}" in self.model_output["beam"] and f"alpha_{plane}" in self.model_output["beam"]:
                            twsv = {
                                "beta": self.model_output["beam"][f"beta_{plane}"][idx],
                                "alpha": self.model_output["beam"][f"alpha_{plane}"][idx],
                            }
                            for emit in [f"ecn{plane}", f"emit_{plane}n", f"nemit_{plane}"]:
                                if emit in self.model_output["beam"]:
                                    twsv.update({"nEmit": self.model_output["beam"][emit][idx]})
                            if plane == "x":
                                bea1.beam.rematchXPlane(**twsv)
                            else:
                                bea1.beam.rematchYPlane(**twsv)
                    gen.write(beam=bea1)
                    if e.name == self.end:
                        self.global_parameters["beam"] = bea1

    @staticmethod
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    @staticmethod
    def find_nearest_idx(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx


# match key:
#                 case "data":
#                     beamdata.update({"data": {}})
#                     for p in dataprops:
#                         try:
#                             f = BytesIO()
#                             np.savez_compressed(f, allow_pickle=False, value=getattr(beam.data, p).val)
#                             f.seek(0)
#
#                             beamdata["data"].update({p: base64.b64encode(f.read()).decode()})
#                         except:
#                             pass
#                 case "emittance":
#                     beamdata.update({"emittance": {}})
#                     for p in emitprops:
#                         try:
#                             f = BytesIO()
#                             np.savez_compressed(f, allow_pickle=False, value=getattr(beam.emittance, p).val)
#                             f.seek(0)
#
#                             beamdata["emittance"].update({p: base64.b64encode(f.read()).decode()})
#                         except:
#                             pass
#                 case "twiss":
#                     beamdata.update({"twiss": {}})
#                     for p in twissprops:
#                         try:
#                             f = BytesIO()
#                             np.savez_compressed(f, allow_pickle=False, value=getattr(beam.twiss, p).val)
#                             f.seek(0)
#
#                             beamdata["twiss"].update({p: base64.b64encode(f.read()).decode()})
#                         except:
#                             pass
#                 case "slice":
#                     beamdata.update({"slice": {}})
#                     for p in sliceprops:
#                         try:
#                             f = BytesIO()
#                             np.savez_compressed(f, allow_pickle=False, value=getattr(beam.slice, p).val)
#                             f.seek(0)
#
#                             beamdata["slice"].update({p: base64.b64encode(f.read()).decode()})
#                         except:
#                             pass
#                 case "sigmas":
#                     beamdata.update({"sigmas": {}})
#                     for p in sigmaprops:
#                         try:
#                             f = BytesIO()
#                             np.savez_compressed(f, allow_pickle=False, value=getattr(beam.sigmas, p).val)
#                             f.seek(0)
#
#                             beamdata["sigmas"].update({p: base64.b64encode(f.read()).decode()})
#                         except:
#                             pass
#                 case "centroids":
#                     beamdata.update({"centroids": {}})
#                     for p in centroidprops:
#                         try:
#                             f = BytesIO()
#                             np.savez_compressed(f, allow_pickle=False, value=getattr(beam.centroids, p).val)
#                             f.seek(0)
#
#                             beamdata["centroids"].update({p: base64.b64encode(f.read()).decode()})
#                         except:
#                             pass
#                 case "kde":
#                     raise NotImplementedError()
#                 case "mve":
#                     raise NotImplementedError()