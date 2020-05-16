import os

import requests

from backend.stream.apis.base import ApiClient


class NewsApiClient(ApiClient):
    ROOT_URL = 'https://newsapi.org/v2/'

    def __init__(self):
        super().__init__('News API')
        self.API_KEY = os.environ.get('NEWS_API_API_KEY', '')

    def fetch(self, endpoint='everything', **kwargs):
        api_url = self.ROOT_URL + endpoint
        params = kwargs.copy()
        for key in list(params.keys()):
            value = params[key]
            if value is None or value == '':
                params.pop(key)

        params['apiKey'] = self.API_KEY
        response = requests.get(url=api_url, params=params)
        return response.json()

    def push(self, bot_text, **kwargs):
        pass
