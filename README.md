# covid19-backend

[![Build Status](https://travis-ci.com/ucinlp/covid19-backend.svg?branch=master)](https://travis-ci.com/ucinlp/covid19-backend)

Code for running all the background services for Covid19 efforts.

## Running locally

To run the server locally you will need to:

1. Set up your virtual environment
```{bash}
conda create -n covid19-backend python=3.7
conda activate covid19-backend
pip install -r requirements.txt
pip install -e .
```

2. Start the server
```{bash}
python app.py
```

## Running via Docker

If the backend is running on the same server as the frontend, or you do not need SSL certification:
```{bash}
docker run -p 2020:2020 \
           -v $HOME/.cache:/root/.cache \
           --rm \
           rloganiv/covid19-backend:latest
```

If you need SSL certification:

```{bash}
docker run -p 2020:2020 \
           -v $HOME/.cache:/root/.cache \
           -v [DIRECTORY CONTAINING SSL CERTS]:/certs \
           --rm \
           rloganiv/covid19-backend:latest \
           -b 0.0.0.0:2020 \
           --certfile /certs/[CERTFILE NAME] \
           --keyfile /certs/[KEYFILE NAME]
```

## Syncing to Google Sheets

For annotation purposes, we want to be able to transfer our data files to Google Sheets. To do that, follow the following steps:

1. Get the `credentials.json` file and copy to root folder, either from Google Drive folder or from [here](https://developers.google.com/sheets/api/quickstart/python).

2. (Make sure you've run `pip install` as above recently)

3. Run the script, which by default copies the `misconceptions.jsonl` file to Google sheet in the shared folder.

```(bash)
python -m backend.utils.jsonl_to_gsheet
```

## Setting up DB
```
python3 -m scripts.yaml2db --config configs/db/source.yaml --db adhoc_backend.db
python3 -m scripts.yaml2db --config configs/db/label.yaml --db adhoc_backend.db
python3 -m scripts.yaml2db --config configs/db/model.yaml --db adhoc_backend.db
```