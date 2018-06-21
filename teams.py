from selenium import webdriver
#from bs4 import BeautifulSoup
#from selenium.webdriver.chrome.options import Options
#import re

#driver = webdriver.Chrome(chrome_options=options, executable_path="C:/Users/eshve/Downloads/chromedriver_win32/chromedriver.exe")
driver = webdriver.Chrome(executable_path="C:/Users/eshve/Downloads/chromedriver_win32/chromedriver.exe")
listofteams = []

with open("C:/Users/eshve/Desktop/teams.csv",'w') as outfile:
    driver.get('https://www.whoscored.com/Regions/247/Tournaments/36/International-FIFA-World-Cup')
    #teams = driver.find_elements_by_xpath('.//*[@class="team-link"]') 
    #teams = driver.find_elements_by_partial_link_text('/Teams/')
    teams = driver.find_elements_by_xpath("//a[@href]")
    for team in teams:
        text = team.get_attribute('href')
        if (text not in listofteams):
            listofteams.append(text)
            output = text + '\n'
            if ('/Teams/' in output):
                outfile.write(output)
                print(output)
outfile.close()
driver.close()