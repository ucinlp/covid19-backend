# Adhoc scripts

## covid19_dataset_downloader.py

### Download tweets related to COVID-19

1. Add Twitter tokens to `env_var.sh`
2. Execute the following commands e.g.,
```
cd
git clone https://github.com/echen102/COVID-19-TweetIDs.git
source configs/env_var.sh
cd covid19-backend
OUTPUT_DIR="./tweet_data/"
python3 -m scripts.stream.covid19_dataset_downloader.py --input ~/COVID-19-TweetIDs/ --output ${OUTPUT_DIR}
```