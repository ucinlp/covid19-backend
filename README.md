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
python3 -m scripts.yaml2db --config configs/db/source.yaml --db backend.db
python3 -m scripts.yaml2db --config configs/db/label.yaml --db backend.db
python3 -m scripts.yaml2db --config configs/db/model.yaml --db backend.db
python3 -m scripts.jsonl2db --input misconceptions.jsonl --table Misinformation --custom initial_wiki --db backend.db
# Note: merged.csv is not available on the repository as it contains tweet texts
python3 -m scripts.csv2db --input merged.csv --tables Input Output --custom old_csv_format --db backend.db
```

Note,
* `merged.csv` consists of 7 columns in the following order with no headers: Random number 1, Misconception ID , Misconception, Tweet, Random Number 2, Annotated Label, Tweet ID

## Train

#### Logistic Regression - Bag of Words
```
python3 -m scripts.ml.train_logreg --train PATH\TO\TRAIN_DATA.jsonl --dev PATH\TO\DEV_DATA.jsonl --output-dir PATH\TO\SAVE\MODEL -- c <INT> --feature-type bow
```
#### Logistic Regression - Avg. GloVE Emebeddings
```
python3 -m scripts.ml.train_logreg --train PATH\TO\TRAIN_DATA.jsonl --dev PATH\TO\DEV_DATA.jsonl --output-dir PATH\TO\SAVE\MODEL --c <INT> --feature-type boe
```
Note, use value of C which provides highest development set accuracy

#### BiLSTM
```
python3 -m scripts.ml.train_bilstm --train PATH\TO\TRAIN_DATA.jsonl --dev PATH\TO\DEV_DATA.jsonl --output-dir PATH\TO\SAVE\MODEL --epochs 20
```
#### SBERT
```
python3 -m scripts.ml.train_nli --model-name bert-base-cased --batch_size=10 --epochs=10 --lr=5e-5 --accumulation_steps 32 --train PATH\TO\TRAIN_DATA.jsonl --dev PATH\TO\DEV_DATA.jsonl --ckpt CKPT_NAME
```
#### SBERT - DA
```
python3 -m scripts.ml.train_nli --model-name digitalepidemiologylab/covid-twitter-bert --batch_size=10 --epochs=10 --lr=5e-5 --accumulation_steps 32 --train PATH\TO\TRAIN_DATA.jsonl --dev PATH\TO\DEV_DATA.jsonl --ckpt CKPT_NAME

```

## Predict
```
python3 -m scripts.ml.predict --model_name MODEL_NAME --model_dir PATH\TO\MODEL --db_input backend.db --file PATH\TO\SAVE\PREDICTIONS.csv
```

Note,
* `MODEL_NAME` for each model can be found in `model_names.txt`
* Predictions can be written to `.csv` file and/or `DB`. Include `--output_dir backend.db` to write predictions to DB.

## Evaluate

```
python3 -m scripts.ml.evaluate --db backend.db --model_name MODEL_NAME --eval_data Arjuna --file_name PATH\TO\PREDICTIONS.csv
```
Note,
* `MODEL_NAME` for each model can be found in `model_names.txt`
* Predictions can be read from `.csv` or `DB`. If `--file_name` not provided then will read predictions for given `MODEL_NAME` from `DB`
