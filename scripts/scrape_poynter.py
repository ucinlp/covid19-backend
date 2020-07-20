from pathlib import Path
import time

from selenium import webdriver
from selenium.webdriver.firefox.options import Options


BASE_URL = 'https://www.poynter.org/ifcn-covid-19-misinformation/page/{i}'
OUTPUT_DIR = Path('poynter_html')


def crawl():
    options = Options()
    options.headless = True
    driver = webdriver.Firefox(options=options)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for i in range(1, 372):
        url = BASE_URL.format(i=i)
        print(url)
        driver.get(url)
        with open(OUTPUT_DIR / f'{i}.html', 'w') as f:
            f.write(driver.page_source)
        time.sleep(5)


if __name__ == '__main__':
    crawl()

