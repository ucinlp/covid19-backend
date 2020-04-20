# covid19-backend

[![Build Status](https://travis-ci.com/ucinlp/covid19-backend.svg?branch=master)](https://travis-ci.com/ucinlp/covid19-backend)

Code for running all the background services for Covid19 efforts.

To run the server you will need to:

1. Set up your virtual environment
```{bash}
conda create -n covid19-backend python=3.7
conda activate covid19-backend
pip install -r requirements.txt
```

2. Start the server
```{bash}
python app.py
```

## Syncing to Google Sheets

For annotation purposes, we want to be able to transfer our data files to Google Sheets. To do that, follow the following steps:

1. Get the `credentials.json` file and copy to root folder, either from Google Drive folder or from [here](https://developers.google.com/sheets/api/quickstart/python).

2. (Make sure you've run `pip install` as above recently)

3. Run the script, which by default copies the `misconceptions.jsonl` file to Google sheet in the shared folder.

```(bash)
python -m backend.utils.jsonl_to_gsheet
```