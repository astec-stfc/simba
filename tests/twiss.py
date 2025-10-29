import sys

sys.path.append("../")
from SimulationFramework.Modules.Twiss import twiss as rtf

twiss = rtf.load_directory(
    directory=r"C:\Users\jkj62.CLRC\Documents\GitHub\integrated-model\restframe\CLARA\246eb190-6660-47bd-b4ba-16b5079e9fab",
    verbose=True,
)
print(twiss["element_name"])
