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

player_index = []
output = []

for league in leagueli:
    url = url_pre + league + url_post
    page = open(url, 'rt', encoding = 'utf-8').read()
    soup = BeautifulSoup(page, 'html.parser') 
    t = soup.find_all(attrs= {"class" : "firstcol"}) 
    
    player_count = 1
    for idx, _ in enumerate(t):
        # modulo 2 단위로 선수에 대한 정보가 들어옴
        if(idx % 2 == 0):
            year_store = []
            team_store = []
            uefa_store = []
            pay_store = []
            player_name = _.contents[1]
            player_index.append(player_name)

            driver.get(url_player + _["href"])
            response = requests.get(url_player + _["href"],headers = header)
            year_team = driver.find_elements(By.CSS_SELECTOR,'.name-column .firstcol')
            gross_pw = driver.find_elements(By.CSS_SELECTOR,'.money-column') # 52곱해야함

            flag = False
            for index, txt in enumerate(year_team):
                if(index % 2 == 0):
                    year_tmp = txt.text
                    # 2023-2024년 이후부터는 자름
                    if(year_tmp > '2022-2023'):
                        flag = False
                        continue
                    else:
                        year_store.append(year_tmp) 
                        flag = True
                elif flag == True:
                    team_tmp = txt.text
                    team_store.append(team_tmp)
            
            for index, txt in enumerate(gross_pw):
                if(index in [0, 1, 2, 6, 7, 8]) : continue
                if(index % 3 == 0):
                    payment = txt.text
                    if(payment != "") :
                        pay_store.append(payment * 52) # 52를 곱해서 연봉이 들어가게끔
        
            # 초과하는 부분들은 crop시킴
            if len(pay_store) > len(year_store):
                pay_store = pay_store[:len(year_store)]

            # 위 과정을 통해 pay, year, team은 오름차순으로 정렬됨
            year_store.reverse()
            team_store.reverse()
            pay_store.reverse()
            
            team_exist = True
            for _ in range(len(year_store)):
                cur_year = year_store[_]
                cur_team = team_store[_]
                if(cur_year != '2022-2023') :
                    if(cur_team in uefa_coefficient.index):
                        uefa_score = uefa_coefficient.loc[cur_team][cur_year]
                        uefa_store.append(uefa_score)
                    else:
                        team_exist = False
                else:
                    uefa_store.append(uefa_coefficient.loc[cur_team]['2021-2022']) #2022-2023년 UEFA coefficient는 2021-2022년 UEFA coefficient를 사용

            # 없는 팀이 존재하는 경우 해당 선수는 무시
            if team_exist == False:
                continue
            
            # 10년치 중에 빈 데이터가 존재하는 경우 : "-"로 채우고 나중에 masking 처리함
            if len(year_store) < 10:
                for _ in range(10 - len(year_store)):
                    year_store.append("-")

            if len(team_store) < 10:
                for _ in range(10 - len(team_store)):
                    team_store.append("-")

            if len(uefa_store) < 10:
                for _ in range(10 - len(uefa_store)):
                    uefa_store.append("-")

            if len(pay_store) < 10:
                for _ in range(10 - len(pay_store)):
                    pay_store.append("-")
        
            for i in range(10):
                output.append(year_store[i])
                output.append(team_store[i])
                output.append(uefa_store[i])
                output.append(pay_store[i])

output = np.array(output).reshape(-1, 40)
output = pd.DataFrame(output, index = player_index)

output.to_csv('./datasets/year,team,uefa,pay.csv', sep = ",")