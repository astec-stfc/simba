import munch

try:
    import sdds
except ImportError:
    print("No SDDS available!")

import numpy as np
import enum


def read_sdds_file(filename, ascii=False, object=None):
    sddsobject = SDDSFile(index=0, ascii=ascii)
    sddsobject.read_file(filename)
    if object is None:
        object = munch.Munch()
        for k, v in sddsobject.data.items():
            object[k] = v
        return object
    return sddsobject


class MyEnumMeta(enum.EnumMeta):
    def __contains__(cls, item):
        try:
            cls(item)
        except ValueError:
            return False
        else:
            return True


class SDDS_Types(enum.IntEnum, metaclass=MyEnumMeta):
    SDDS_LONGDOUBLE = 1
    SDDS_DOUBLE = 2
    SDDS_REAL64 = 2
    SDDS_FLOAT = 3
    SDDS_REAL32 = 3
    SDDS_LONG = 4
    SDDS_INT32 = 4
    SDDS_ULONG = 5
    SDDS_UINT32 = 5
    SDDS_SHORT = 6
    SDDS_INT16 = 6
    SDDS_USHORT = 7
    SDDS_UINT16 = 7
    SDDS_STRING = 8
    SDDS_CHARACTER = 9
    SDDS_NUM_TYPES = 9
    SDDS_BINARY = 1
    SDDS_ASCII = 2


class SDDSObject(munch.Munch):

    def __init__(
        self,
        index=1,
        name=None,
        data=[],
        unit="",
        type=2,
        symbol="",
        formatstring="",
        fieldlength=0,
        description="",
    ):
        super().__init__()
        self._types = SDDS_Types
        self._name = name
        self._data = data
        self._unit = unit
        self._symbol = symbol
        self._type = type
        self._formatstring = formatstring
        self._fieldlength = fieldlength
        self._description = description

    def __repr__(self):
        return repr(
            {
                "name": self.name,
                "unit": self.unit,
                "type": self._types(self.type).name,
                "data": self.data,
            }
        )

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name
        return self._name

    @property
    def unit(self):
        return self._unit

    @unit.setter
    def unit(self, unit):
        if unit is not None:
            self._unit = unit
        return self._unit

    @property
    def symbol(self):
        return self._symbol

    @symbol.setter
    def symbol(self, symbol):
        if symbol is not None:
            self._symbol = symbol
        return self._symbol

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, type):
        if isinstance(type, str):
            if type in self._types:
                self._type = type
        elif isinstance(type, int):
            self._type = type
        return self._type

    @property
    def fieldlength(self):
        return self._fieldlength

    @fieldlength.setter
    def fieldlength(self, length):
        if isinstance(length, (int, float)):
            self._fieldlength = int(fieldlength)
        return self._fieldlength

    @property
    def formatstring(self):
        return self._formatstring

    @formatstring.setter
    def formatstring(self, string):
        if isinstance(string, str):
            self._formatstring = string
        return self._formatstring

    @property
    def description(self):
        return self._description

    @description.setter
    def description(self, string):
        if isinstance(string, str):
            self._description = string
        return self._description

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data
        return self._data


class SDDSColumn(SDDSObject):

    def __init__(
        self,
        name=None,
        data=[],
        unit="",
        type=2,
        symbol="",
        formatstring="",
        fieldlength=0,
        description="",
    ):
        super().__init__(name=name, data=None, unit=unit, type=type, symbol=symbol)
        self.objectType = "Column"
        self.data = data

    @property
    def data(self):
        if isinstance(self._data, (np.ndarray)):
            return self._data.tolist()
        return self._data

    @data.setter
    def data(self, data):
        if not isinstance(data, (dict, tuple, list, np.ndarray)):
            if isinstance(data, (float, int, str)):
                data = [data]
            else:
                raise Exception("Wrong data type for SDDS Column!", type(data))
        self._data = data
        return self._data

    def length(self):
        return len(self._data)


class SDDSParameter(SDDSObject):

    def __init__(
        self,
        name=None,
        data=[],
        unit="",
        type=2,
        symbol="",
        formatstring="",
        fieldlength=0,
        description="",
    ):
        super().__init__(name=name, data=None, unit=unit, type=type, symbol=symbol)
        self.objectType = "Parameter"
        self.data = data

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        if not isinstance(data, (dict, tuple, list)):
            self._data = [data]
        else:
            raise Exception("Wrong data type for SDDS Parameter!", type(data))
        return self._data

    @property
    def fieldlength(self):
        if self._fieldlength != 0:
            return self._fieldlength
        else:
            return ""


class SDDSArray(munch.Munch):

    def __init__(self, columns):
        super().__init__(columns)

    def __getitem__(self, itemkey):
        try:
            return {
                k: getattr(v, itemkey) if hasattr(v, itemkey) else v[itemkey]
                for k, v in self.items()
            }
        except:
            try:
                return super().__getitem__(itemkey)
            except KeyError:
                raise AttributeError(itemkey)


class SDDSFile(object):

    def __init__(self, index=1, ascii=False):
        super().__init__()
        self._types = SDDS_Types
        self._columns = munch.Munch()
        self._parameters = munch.Munch()
        if index is None:
            index = 0
        self._index = index
        try:
            self._sddsObject = sdds.SDDS(self._index)
        except:
            self._sddsObject = sdds.sdds.SDDS(self._index)
        if ascii:
            self._sddsObject.mode = self._sddsObject.SDDS_ASCII
        else:
            self._sddsObject.mode = self._sddsObject.SDDS_BINARY

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, index):
        self._index = index
        return self._index

    def clear(self):
        self._columns = munch.Munch()
        self._parameters = munch.Munch()
        try:
            self._sddsObject = sdds.SDDS(None)
        except:
            self._sddsObject = sdds.sdds.SDDS(None)

    def column_names(self):
        return self._columns.keys()

    def parameter_names(self):
        return self._parameters.keys()

    def columns(self):
        return SDDSArray(self._columns)

    def parameters(self):
        return SDDSArray(self._parameters)

    def add_column(
        self,
        name,
        data,
        type=2,
        unit="",
        symbol="",
        formatstring="",
        fieldlength=0,
        description="",
    ):
        self._columns[name] = SDDSColumn(
            name=name,
            data=data,
            unit=unit,
            type=type,
            symbol=symbol,
            formatstring=formatstring,
            fieldlength=fieldlength,
            description=description,
        )

    def add_columns(
        self,
        name,
        data,
        type,
        unit,
        symbol,
        formatstring=None,
        fieldlength=None,
        description=None,
    ):
        combined_data = {
            k: v
            for k, v in {
                "name": name,
                "data": data,
                "type": type,
                "unit": unit,
                "symbol": symbol,
                "formatstring": formatstring,
                "fieldlength": fieldlength,
                "description": description,
            }.items()
            if v is not None
        }
        for i in range(len(name)):
            data = dict(
                zip(
                    combined_data.keys(),
                    [combined_data[k][i] for k in combined_data.keys()],
                )
            )
            self.add_column(**data)

    def add_parameter(
        self,
        name,
        data,
        type=2,
        unit="",
        symbol="",
        formatstring="",
        fieldlength=0,
        description="",
    ):
        self._parameters[name] = SDDSParameter(
            name=name,
            data=data,
            unit=unit,
            type=type,
            symbol=symbol,
            formatstring=formatstring,
            fieldlength=fieldlength,
            description=description,
        )

    def add_parameters(
        self,
        name,
        data,
        type,
        unit,
        symbol,
        formatstring=None,
        fieldlength=None,
        description=None,
    ):
        combined_data = {
            k: v
            for k, v in {
                "name": name,
                "data": data,
                "type": type,
                "unit": unit,
                "symbol": symbol,
                "formatstring": formatstring,
                "fieldlength": fieldlength,
                "description": description,
            }.items()
            if v is not None
        }
        for i in range(len(name)):
            data = dict(
                zip(
                    combined_data.keys(),
                    [combined_data[k][i] for k in combined_data.keys()],
                )
            )
            self.add_parameter(**data)

    def save(self, *args, **kwargs):
        return self.write_file(*args, **kwargs)

    def dimensions(self, list):
        l = list
        for i in range(5):
            try:
                l = l[0]
            except:
                return i

    def write_file(self, filename):
        for name, param in self._parameters.items():
            self._sddsObject.defineParameter(
                param.name,
                param.symbol,
                param.unit,
                param.description,
                param.formatstring,
                param.type,
                param.fieldlength,
            )
            self._sddsObject.setParameterValueList(param.name, param.data)
        for name, column in self._columns.items():
            # print(len([list(column.data)][0]))
            self._sddsObject.defineColumn(
                column.name,
                column.symbol,
                column.unit,
                column.description,
                column.formatstring,
                column.type,
                column.fieldlength,
            )
            self._sddsObject.setColumnValueLists(column.name, [column.data])
        self._sddsObject.save(filename)

    def load(self, *args, **kwargs):
        return self.read_file(*args, **kwargs)

    def read_file(self, filename, page=-1):
        self._sddsObject.load(filename)
        sddsref = self._sddsObject
        for col in range(len(sddsref.columnName)):
            symbol, unit, description, formatString, type, fieldLength = (
                sddsref.columnDefinition[col]
            )
            column_data = np.array(sddsref.columnData[col][page])
            self.add_column(
                sddsref.columnName[col],
                column_data,
                type=type,
                unit=unit,
                symbol=symbol,
                formatstring=formatString,
                fieldlength=fieldLength,
                description=description,
            )
        # sddsobject.SDDSparameterNames = list()
        for param in range(len(sddsref.parameterName)):
            name = sddsref.parameterName[param]
            symbol, unit, description, formatString, type, fieldLength = (
                sddsref.parameterDefinition[param]
            )
            parameter_data = sddsref.parameterData[param]
            self.add_parameter(
                sddsref.parameterName[param],
                parameter_data[page],
                type=type,
                unit=unit,
                symbol=symbol,
                formatstring=formatString,
                fieldlength=fieldLength,
                description=description,
            )

    @property
    def data(self):
        return {k: v.data for k, v in {**self._parameters, **self._columns}.items()}
