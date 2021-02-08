

class Feature():
    '''

    '''

    def __init__(self, client, name=None, sql=None, featureType=None, startTime=None, windowInterval=None):
        self.client = client
        self.id = None
        self.name = name
        self.sql = sql
        self.feature_type = featureType
        self.start_time = startTime
        self.window_interval = windowInterval

    def __repr__(self):
        return f"Feature(name={repr(self.name)}, sql={repr(self.sql)}, feature_type={repr(self.feature_type)}, start_time={repr(self.start_time)}, window_interval={repr(self.window_interval)})"

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.id == other.id

    def to_dict(self):
        return {'name': self.name, 'sql': self.sql, 'feature_type': self.feature_type, 'start_time': self.start_time, 'window_interval': self.window_interval}
