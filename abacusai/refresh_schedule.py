

class RefreshSchedule():
    '''
        A refresh schedule for an object. Defines when the next version of the object will be created
    '''

    def __init__(self, client, refreshPolicyId=None, nextRunTime=None, cron=None, refreshType=None):
        self.client = client
        self.id = None
        self.refresh_policy_id = refreshPolicyId
        self.next_run_time = nextRunTime
        self.cron = cron
        self.refresh_type = refreshType

    def __repr__(self):
        return f"RefreshSchedule(refresh_policy_id={repr(self.refresh_policy_id)}, next_run_time={repr(self.next_run_time)}, cron={repr(self.cron)}, refresh_type={repr(self.refresh_type)})"

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.id == other.id

    def to_dict(self):
        return {'refresh_policy_id': self.refresh_policy_id, 'next_run_time': self.next_run_time, 'cron': self.cron, 'refresh_type': self.refresh_type}
