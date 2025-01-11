from autoscraper import AutoScraper
import pandas as pd

##https://github.com/deepcharts/projects/blob/main/AutoScraper%20Tutorial.ipynb
## example: https://oxylabs.io/blog/automated-web-scraper-autoscraper

"""
AutoScraper is a web scraping library written in Python3; it’s known for being lightweight, intelligent, and easy to use – even beginners can use it without an in-depth understanding of a web scraping. 

AutoScraper accepts the URL or HTML of any website and scrapes the data by learning some rules. In other words, it matches the data on the relevant web page and scrapes data that follow similar rules.
"""
# Web Page to Scrape from
url = "https://www.noaa.gov/media-releases"

# Example Text to Pull
# Note: Change below to most recent news release headline on 'https://www.noaa.gov/media-releases'
news_list = [
    "Applications now open nationwide for community-led heat-monitoring campaigns"
]

# Initialize AutoScraper
scraper = AutoScraper()

# Build Model
news_result = scraper.build(url, news_list)

# Review Results
news_result


print("================================================================")
# Web Page to Scrape from
url = "https://en.wikipedia.org/wiki/List_of_counties_in_California"

# Example Text to Pull
county_list = ["Alameda County", "Yuba County"]
est_list = ["1,622,188", "85,722"]


# Initialize AutoScraper
scraper = AutoScraper()

# Build Model
county_result = scraper.build(url, county_list)
est_result = scraper.build(url, est_list)

# Review Results
print(county_result)
print(est_result)

print("================================================================")

# Convert Lists to Dictionary
data = {"County": county_result, "Estimated Population": est_result}

# Convert Dictionary to Dataframe
df = pd.DataFrame(data)

df
