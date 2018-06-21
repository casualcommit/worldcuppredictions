from selenium import webdriver
import os
from bs4 import BeautifulSoup
from selenium.webdriver.chrome.options import Options
import re
import time

#driver = webdriver.Chrome(chrome_options=options, executable_path="C:/Users/eshve/Downloads/chromedriver_win32/chromedriver.exe")
driver = webdriver.Chrome(executable_path="C:/Users/eshve/Downloads/chromedriver_win32/chromedriver.exe")

for filename in os.listdir('C:/Users/eshve/Desktop/Teams'):
    with open('C:/Users/eshve/Desktop/Teams/'+filename,'r') as infile, open('C:/Users/eshve/Desktop/PlayersTeams/'+filename,'w') as outfile:
        #driver = webdriver.PhantomJS('C:/Users/eshve/Downloads/phantomjs-2.1.1-windows/phantomjs-2.1.1-windows/bin/phantomjs.exe')
        for line in infile:
            #options = Options()
            #options.set_headless(headless=True)
            #driver = webdriver.Firefox(executable_path='C:/Users/eshve/Downloads/geckodriver-v0.20.1-win64/geckodriver.exe')
            #options = Options()
            #options.set_headless(headless=True)
            #driver = webdriver.Chrome(chrome_options=options, executable_path="C:/Users/eshve/Downloads/chromedriver_win32/chromedriver.exe")
            driver.get(line)
            names = driver.find_elements_by_xpath('.//*[@class="player-info-block"]')
            ratings = driver.find_elements_by_xpath('.//*[@class="rating"]')
            try:
                text = names[0].get_attribute('innerHTML')
                result = re.findall("<dd>(.*?)</dd>", text)
            except:
                result = 'empty'
            try:  
                text2 = ratings[0].get_attribute('innerHTML')
                result2 = text2.replace('<strong>','').replace('</strong>','')
                result2 = "".join(result2.split())
                result2 = result2.replace(r'<spanclass="not-available">N/A</span>',"0.00")
            except:
                result2 = 'empty'
            output = result[0] + ', ' + result2 + '\n'
            #driver.close()
            outfile.write(output)
            print(line)
            time.sleep(2)
    outfile.close()