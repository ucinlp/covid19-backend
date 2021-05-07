# Adhoc scripts

## covid19_dataset_downloader.py

### Download tweets related to COVID-19

1. Add Twitter tokens to `env_var.sh`
2. Execute the following commands e.g.,
```
cd
git clone https://github.com/echen102/COVID-19-TweetIDs.git
. configs/env_var.sh
cd covid19-backend
OUTPUT_DIR="./tweet_data/"
python3 -m scripts.stream.covid19_dataset_downloader.py --input ~/COVID-19-TweetIDs/ --output ${OUTPUT_DIR}
```

## tweet_crawler.py

### Crawling tweets related to HIV

1. Add Twitter tokens to `env_var.sh`
2. Execute the following commands e.g.,
```
. configs/env_var.sh
python3 -m scripts.stream.tweet_crawler.py --config configs/twitter/hiv.yaml --output ${OUTPUT_DIR}
```

## tweet_counter.py

### Count downloaded tweets (Numbers of unique tweets, users, etc)
Execute the following commands
```
python3 -m scripts.stream.tweet_counter.py --input ${INPUT_DIR}
```
