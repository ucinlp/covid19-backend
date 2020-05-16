class ApiClient(object):
    def __init__(self, service):
        self.service = service

    def fetch(self, *args, **kwargs):
        raise NotImplementedError

    def push(self, bot_text, **kwargs):
        raise NotImplementedError
