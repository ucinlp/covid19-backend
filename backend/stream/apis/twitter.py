import requests
import os

from backend.stream.apis.base import ApiClient


class TwitterClient(ApiClient):
    ROOT_URL = 'https://api.twitter.com/1.1/'

    def __init__(self):
        super().__init__('Twitter')
        self.CONSUMER_KEY = os.environ.get('', None)
        self.CONSUMER_SECRET = os.environ.get('', None)
        self.ACCESS_TOKEN = os.environ.get('', None)
        self.ACCESS_TOKEN_SECRET = os.environ.get('', None)

    def fetch(self, query, endpoint='everything', country=None):
        params = {'q': query}
        response = requests.get(url=self.ROOT_URL + endpoint, params=params)
        return response.json()

    def fetch_selected(self):
        pass

    def push(self, bot_text, **kwargs):
        pass
