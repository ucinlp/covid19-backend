from setuptools import setup, find_packages


with open('README.md', 'r') as f:
    long_description = f.read()


setup(
    name='covid19-backend',
    version='0.0.1',
    description='Backend for UCI COVID-19 misinformation detection project.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/ucinlp/covid19-backend',
    packages=find_packages(exclude=('tests', 'scripts')),
    python_requires='>=3.6',
    install_requires=[
        'flask>=1.0.2',
        'flask-cors>=3.0.0',
        'google-api-python-client',
        'google-auth-httplib2',
        'google-auth-oauthlib',
        'gunicorn>=20.0.4',
        'transformers[torch]==2.7.0',  # TODO: Upgrade to version 3.
        'sqlalchemy>=1.3.16'
    ],
    extras_require={
        'test': ['flake8', 'pytest']
    },
)
