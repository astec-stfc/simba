import os
import uuid
import shutil

class TemporaryDirectory(object):
    """Context manager for tempfile.mkdtemp() so it's usable with "with" statement."""

    def __init__(self, dir=os.getcwd(), *args, **kwargs):
        self.dir = dir
        self.args = args
        self.kwargs = kwargs

    def tempname(self):
        return "tmp" + str(uuid.uuid4())

    def __enter__(self):
        exists = True
        while exists:
            self.name = self.dir + "/" + self.tempname()
            if not os.path.exists(self.name):
                exists = False
                os.makedirs(self.name)
        return self.name

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            shutil.rmtree(self.name)
        except Exception:
            pass