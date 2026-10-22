import os

if "OMPI_COMM_WORLD_SIZE" not in os.environ:
    try:
        import mpi4py

        mpi4py.rc.initialize = False
    except ImportError:
        pass
