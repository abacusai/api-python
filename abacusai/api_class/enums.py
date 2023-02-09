from enum import Enum


class ApiEnum(Enum):
    def __eq__(self, other):
        if isinstance(other, str):
            return self.value.upper() == other.upper()
        elif isinstance(other, int):
            return self.value == other
        elif other is None:
            return self.value == ''
        return super().__eq__(other)

    def __hash__(self):
        if isinstance(self.value, str):
            return hash(self.value.upper())
        return hash(self.value)


class SamplingMethodType(ApiEnum):
    N_SAMPLING = 'N_SAMPLING'
    PERCENT_SAMPLING = 'PERCENT_SAMPLING'
