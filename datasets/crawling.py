from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

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
url_pre = "./datasets/league/"
url_post = ".html"

url_player = "https://capology.com"

uefa_coefficient = pd.read_csv("datasets/uefa/uefa_result.csv", index_col = 0)
# 출력 예시 : print(uefa_coefficient.loc["Real Madrid"]["2012-2013"])

for league in leagueli:
    url = url_pre + league + url_post
    page = open(url, 'rt', encoding = 'utf-8').read()
    soup = BeautifulSoup(page, 'html.parser') 
    t = soup.find_all(attrs= {"class" : "firstcol"}) 
    
    player_count = 1
    for idx, _ in enumerate(t):
        if(idx % 2 == 0):
            pay_store = []
            year_store = []
            team_store = []
            uefa_store = []
            performance_store = []
            
            player_name = _.contents[1]
            driver.get(url_player + _["href"])
            response = requests.get(url_player + _["href"],headers = header)
            year_team = driver.find_elements(By.CSS_SELECTOR,'.name-column .firstcol')
            gross_pw = driver.find_elements(By.CSS_SELECTOR,'.money-column')

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

        # 위 과정을 통해 pay, year, team은 오름차순으로 정렬됨
        pay_store.reverse()
        year_store.reverse()
        team_store.reverse()
            
        for _ in range(len(year_store)):
            cur_year = year_store[_]
            cur_team = team_store[_]
            if(cur_year != '2022-2023') :
                uefa_score = uefa_coefficient.loc[cur_team][cur_year]
                uefa_store.append(uefa_score)
            else:
                uefa_store.append(uefa_coefficient.loc[cur_team]['2021-2022']) #2022-2023년 UEFA coefficient는 2021-2022년 UEFA coefficient를 사용

        # 여기다가 선수 능력치만 넣어주면 됨
        player_url = "https://fbref.com/en/search/search.fcgi?search="

        res = requests.get(player_url + player_name)
        soup = BeautifulSoup(res.content, 'html.parser').contents[3].contents[3].contents[1].contents[21].contents[15].contents[5].contents[1].contents[7]

        year_idx = 0
        for idx, table_value in enumerate(soup):
            if(idx % 2 == 0):
                continue
                
            # 선수 별로 performance : goal, assist, non_penalty_gal, penalty_kick_made, yello_card, red_card
            # 총 6개
            if(table_value.contents[0].contents[0] == year_store[year_idx]):
                year_idx += 1
                goal = table_value.contents[10].contents[0]
                assist = table_value.contents[11].contents[0]
                non_penalty_goal = table_value.contents[12].contents[0]
                penalty_kick_made = table_value.contents[13].contents[0]
                yello_card = table_value.contents[14].contents[0]
                red_card = table_value.contents[15].contents[0]

                performance_store.append(goal)
                performance_store.append(assist)
                performance_store.append(non_penalty_goal)
                performance_store.append(penalty_kick_made)
                performance_store.append(yello_card)
                performance_store.append(red_card)
            else:
                continue
        
        # 해당 선수별로 csv파일이 만들어짐
        # 각 년도별로 다음과 같은 순서로 저장됨
        # uefa score, performance score(performance : goal, assist, non_penalty_gal, penalty_kick_made, yello_card, red_card), pay_store

        # pay_store은 label이므로 따로 빼도 될 것 같음
        result = []

        for year in range(len(year)):
            result.append(uefa_store[year])
            for _ in range(6):
                result.append(performance_store[year * 6 + _])
            result.append(pay_store[year])

        # 만약 10년치 데이터가 전부 존재하지 않는 경우 : NaN으로 집어넣어주기 위함
        # 일종의 masking 처리하는 것
        if(len(year) != 10):
            for i in range(10 - len(year)):
                for j in range(8):
                    result.append("-")
    
        result = np.array(result).reshape(10, -1)
        result = pd.DataFrame(result)
        result.to_csv("./datasets/" + str(player_count) + ".csv", sep = ',')
        player_count += 1