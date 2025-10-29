from munch import Munch
import easygdf


class gdf_emit(Munch):

    def __init__(self, filename) -> None:
        super().__init__(self)
        self._raw_gdf = easygdf.load(filename)
        self._create_emit_dictionary()

    def _create_emit_dictionary(self):
        self.update({b["name"]: b["value"] for b in self._raw_gdf["blocks"]})

    def get_property(self, property):
        if property in self:
            return self[property]
        return None
