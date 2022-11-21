from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import requests
from bs4 import BeautifulSoup
import pandas as pd

header = {
    'User-Agent': 'Mozilla/5.0'
}
chrome_options = Options()
chrome_options.add_argument('headless')
chrome_options.add_experimental_option('detach',True)

chrome_options.add_experimental_option("excludeSwitches",['enable-logging'])

service = Service(executable_path = ChromeDriverManager().install())
driver = webdriver.Chrome(service = service, options = chrome_options)


driver.implicitly_wait(3)
driver.maximize_window()

leagueli = ['premier_league','serie_a',"la_liga",'bundesliga','ligue_1','eredivisie']
url_pre = "./datasets/"
url_post = ".html"

url_player = "https://capology.com"



for league in leagueli:
    url = url_pre + league + url_post
    page = open(url, 'rt', encoding = 'utf-8').read()
    soup = BeautifulSoup(page, 'html.parser') 
    t = soup.find_all(attrs= {"class" : "firstcol"})   
    for idx, _ in enumerate(t):
        if(idx % 2 == 0):
            driver.get(url_player + _["href"])
            response = requests.get(url_player + _["href"],headers = header)
            year_team = driver.find_elements(By.CSS_SELECTOR,'.name-column .firstcol')
            gross_pw = driver.find_elements(By.CSS_SELECTOR,'.money-column')
            
            pay_store = []
            year_store = []
            team_store = []

            for index, txt in enumerate(year_team):
                if(index % 2 == 0):
                    year_tmp = txt.text
                    year_store.append(year_tmp)
                else:
                    team_tmp = txt.text
                    team_store.append(team_tmp)
            
            for index, txt in enumerate(gross_pw):
                if(index in [0, 1, 2, 6, 7, 8]) : continue
                if(index % 3 == 0):
                    payment = txt.text
                    if(payment != "") :
                        pay_store.append(payment)