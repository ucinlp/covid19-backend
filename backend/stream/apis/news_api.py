import os

import requests

from backend.stream.apis.base import ApiClient


class NewsApiClient(ApiClient):
    ROOT_URL = 'https://newsapi.org/v2/'

    def __init__(self):
        super().__init__('News API')
        self.API_KEY = os.environ.get('', None)

    def fetch(self, endpoint='everything', **kwargs):
        api_url = self.ROOT_URL + endpoint
        params = kwargs.copy()
        params['apiKey'] = self.API_KEY
        response = requests.get(url=api_url, params=params)
        return response.json()

    def push(self, bot_text, **kwargs):
        pass
