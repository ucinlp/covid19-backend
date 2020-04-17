import os

import requests

from backend.stream.apis.base import ApiClient


class DiffbotArticleClient(ApiClient):
    ROOT_URL = 'https://api.diffbot.com/v3/article'

    def __init__(self):
        super().__init__('Diffbot Article')
        self.DEV_TOKEN = os.environ.get('', None)

    def fetch(self, article_url, **kwargs):
        params = kwargs.copy()
        params['url'] = article_url
        params['token'] = self.DEV_TOKEN
        response = requests.get(url=self.ROOT_URL, params=params)
        return response.json()

    def push(self, bot_text, **kwargs):
        pass
