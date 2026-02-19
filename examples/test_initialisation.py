import sys
import os

sys.path.append("../")
sys.path.append(r"C:\Users\jkj62.CLRC\Documents\GitHub\laura")
import simba.Framework as fw

framework = fw.Framework(
    directory="testing",
    master_lattice="../../laura-lattices/CLARA",
    generator_defaults="clara.yaml",
    clean=False,
    verbose=True,
)
framework.loadSettings("Lattices/CLARA400_v13_combined.def")
# Change all lattice codes to ASTRA/Elegant/GPT with exclusions (injector cannot be done in Elegant)
framework.change_Lattice_Code("All", "ASTRA", exclude=["Linac"])
# Again, but put the VBC in Elegant for CSR
framework.change_Lattice_Code("FEBE", "Elegant")
# This is the code that generates the laser distribution (ASTRA or GPT)
framework.change_generator("ASTRA")
# Load a starting laser distribution setting
framework.generator.load_defaults("clara_400_2ps_Gaussian")
# Set the thermal emittance for the generator
framework.generator.thermal_emittance = 0.0005
# This is a scaling parameter
# This defines the number of particles to create at the gun (this is "ASTRA generator" which creates distributions)
framework.generator.number_of_particles = 512