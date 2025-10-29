from ocelot import * 

# Drifts
fodo_drift_01 = Drift(l=0.05, eid='FODO_DRIFT_01')

# Quadrupoles
quad1 = Quadrupole(l=0.1, k1=-20.0, eid='QUAD1')
quad2 = Quadrupole(l=0.1, k1=40.0, eid='QUAD2')
quad3 = Quadrupole(l=0.1, k1=-10.0, eid='QUAD3')

# Markers
begin = Marker(eid='BEGIN')
end = Marker(eid='END')

# Lattice 
cell = (begin, quad1, fodo_drift_01, quad2, fodo_drift_01, quad3, end)
