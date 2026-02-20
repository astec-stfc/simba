from ...Framework_objects import frameworkLattice
from ...Modules import Beams as rbf
import os
import json, requests
from ...FrameworkHelperFunctions import expand_substitution
from confluent_kafka import Consumer, Message
import time, uuid
from pydantic import BaseModel
import numpy as np
import inspect
from io import BytesIO
import base64
from copy import deepcopy
from ocelot.cpbd.beam import Twiss
from typing import List


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

    twiss_data: List = []

    class ModelParams(BaseModel):
        """
        Model parameters from the polylithic fastapi
        """
        lattice_section: str
        """Lattice section in which to execute the model"""

        beam_properties: list[str]
        """List of beam properties passed to the model"""

        machine_settings: list[str]
        """List of machine settings required by the model"""

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

    # def endScreen(self, **kwargs):
    #     """
    #     Make the final position in the lattice a
    #     :class:`~SimulationFramework.Elements.screen.screen` object.
    #
    #     Returns
    #     -------
    #     :class:`~SimulationFramework.Elements.screen.screen`
    #         The final screen in the lattice
    #
    #     """
    #     return screen(
    #         name=self.endObject.objectname,
    #         type="screen",
    #         centre=self.endObject.centre,
    #         position_start=self.endObject.position_start,
    #         position_end=self.endObject.position_start,
    #         global_rotation=self.endObject.global_rotation,
    #         global_parameters=self.global_parameters,
    #         **kwargs,
    #     )

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
        tws = Twiss()
        bea = deepcopy(self.global_parameters["beam"])
        tws.xx = bea.sigmas.sigma_x
        tws.yy = bea.sigmas.sigma_y
        tws.tautau = bea.sigmas.sigma_z
        tws.s = self.startObject.middle.z
        self.twiss_data.append(tws)

    def postProcess(self):
        """
        Save the beam file(s) from the polylithic output into HDF5 format
        """
        super().postProcess()
        HDF5filename = self.end + ".openpmd.hdf5"
        rbf.openpmd.write_openpmd_beam_file(
            self.global_parameters["beam"],
            self.global_parameters["master_subdir"] + "/" + HDF5filename,
        )
        tws = Twiss()
        bea = deepcopy(self.global_parameters["beam"])
        tws.xx = bea.sigmas.sigma_x
        tws.yy = bea.sigmas.sigma_y
        tws.s = self.endObject.physical.middle.z
        tws.tautau = bea.sigmas.sigma_z
        self.twiss_data.append(tws)
        twsdat = {e: [] for e in self.twiss_data[0].__dict__.keys()}
        for t in self.twiss_data:
            for k, v in t.__dict__.items():
                # Offset the s values to the start of the lattice
                if k == "s":
                    v += self.startObject.physical.start.z
                twsdat[k].append(v)
        np.savez_compressed(
            f'{self.global_parameters["master_subdir"]}/{self.objectname}_twiss.npz',
            **twsdat,
        )

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
            if debug_no_remote_server:
                model_requirements = self.ModelParams(
                    lattice_section=self.objectname,
                    beam_properties=['data'],
                    machine_settings=[],
                )
            else:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                json_string = response.text
                model_requirements = self.ModelParams.model_validate_json(json_string)
                # if self.objectname != model_requirements.lattice_section:
                #     raise ValueError("Specified model does not match the lattice section, did you use the wrong model?")
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to query polylithic FastAPI: {e}")

        # format beam for requested model data
        beam: rbf.beam = self.global_parameters["beam"]
        beamdata = {}
        emitprops = [name for name, value in inspect.getmembers(type(beam.emittance)) if isinstance(value, property)]
        dataprops = ['x', 'cpx', 'y', 'cpy', 'z', 'cpz']
        sliceprops = [name for name, value in inspect.getmembers(type(beam.slice)) if isinstance(value, property)]
        sigmaprops = [name for name, value in inspect.getmembers(type(beam.sigmas)) if isinstance(value, property)]
        centroidprops = [name for name, value in inspect.getmembers(type(beam.centroids)) if
                         isinstance(value, property)]
        twissprops = [name for name, value in inspect.getmembers(type(beam.twiss)) if isinstance(value, property)]

        beamcopy = deepcopy(self.global_parameters["beam"])
        for key in model_requirements.beam_properties:
            splitkey = key.split(':')
            if len(splitkey) == 2:
                val = getattr(getattr(beamcopy, splitkey[0]), splitkey[1]).val
                if val.ndim == 0:
                    val = float(val)
                beamdata.append(val)
            elif len(splitkey) == 3:
                beamattr = getattr(beamcopy, splitkey[0])
                arg = [getattr(beamcopy, beamattr) for a in splitkey[2].split(',')]
                val = getattr(beamattr, splitkey[1])(arg[0], arg[1]).val
                if val.ndim == 0:
                    val = float(val)
                beamdata.append(val)


        # send kafka the model name, the formatted beam, lattice, and lattice settings
        url = f"{self.executables.polylithic_url}/submit_lattice/{self.model}"
        job_payload = {
            "model": self.model,
            "beam": {},
            "lattice_name": self.objectname,
            "job_id": self.model + uuid.uuid4().hex,
            "lattice": {"lattice": "example"}
        }
        if not debug_no_remote_server:
            try:
                response = requests.post(url, json=job_payload, timeout=10)
                response.raise_for_status()
            except requests.RequestException as e:
                raise RuntimeError(f"Failed to schedule job on polylithic FastAPI: {e}")

            consumer: Consumer = Consumer({'bootstrap.servers': self.executables.kafka_url, 'group.id': "simframe"})
            consumer.subscribe(['pl_job_results'])
            # wait for consumer to return
            start = time.time()
            while True:
                msg: Message = consumer.poll(1.0)
                print("poll")
                if time.time() - start > 60:
                    raise TimeoutError("Timeout while waiting for kafka job")
                if msg is None:
                    continue
                if msg.error():
                    raise ValueError(f"Polylithic kafka consumer error: {msg.error()}")

                msgvalue = msg.value().decode('utf-8')
                print('Received message: {}'.format(msgvalue))
                # convert msgvalue to json

                msgvalue = json.loads(msgvalue)
                if msgvalue["job_id"] == job_payload["job_id"]:
                    print("job complete")
                    break
            consumer.close()

            url = f"{self.executables.polylithic_url}/get_result/{job_payload["job_id"]}"
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                return_json = response.json()
            except requests.RequestException as e:
                raise RuntimeError(f"Failed to query result from polylithic FastAPI: {e}")
        else:
            return_json = job_payload  # for testing purposes, just return the job payload

        return_beam = deepcopy(self.global_parameters["beam"])

        for key in return_json["beam"]:
            match key:
                case "data":
                    for p in dataprops:
                        try:
                            setattr(return_beam.beam, p, np.load(BytesIO(base64.b64decode(return_json["data"][p]))))
                        except:
                            pass
                case "emittance":
                    for p in emitprops:
                        try:
                            setattr(return_beam.emittance, p,
                                    np.load(BytesIO(base64.b64decode(return_json["emittance"][p]))))
                        except:
                            pass
                case "twiss":
                    for p in twissprops:
                        try:
                            setattr(return_beam.twiss, p, np.load(BytesIO(base64.b64decode(return_json["twiss"][p]))))
                        except:
                            pass
                case "slice":
                    for p in sliceprops:
                        try:
                            setattr(return_beam.slice, p, np.load(BytesIO(base64.b64decode(return_json["slice"][p]))))
                        except:
                            pass
                case "sigmas":
                    for p in sigmaprops:
                        try:
                            setattr(return_beam.sigmas, p, np.load(BytesIO(base64.b64decode(return_json["sigmas"][p]))))
                        except:
                            pass
                case "centroids":
                    for p in centroidprops:
                        try:
                            setattr(return_beam.centroids, p,
                                    np.load(BytesIO(base64.b64decode(return_json["centroids"][p]))))
                        except:
                            pass
                case "kde":
                    raise NotImplementedError()
                case "mve":
                    raise NotImplementedError()

        self.global_parameters["beam"] = return_beam


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