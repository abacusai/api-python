class AbstractApiClass:
    def __init__(self, client, id):
        self.client = client
        if id is not None:
            self.id = id

    def __eq__(self, other):
        if self.__class__ == other.__class__:
            if hasattr(self, 'id'):
                return self.id == other.id
            else:
                return self.__dict__ == other.__dict__
        return False

    def _get_attribute_as_dict(self, attribute):
        if isinstance(attribute, list):
            return [elem.to_dict() for elem in attribute if elem]
        elif attribute is not None:
            return attribute.to_dict()
