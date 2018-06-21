from selenium import webdriver
import time
#from bs4 import BeautifulSoup
#from selenium.webdriver.chrome.options import Options
#import re

#driver = webdriver.Chrome(chrome_options=options, executable_path="C:/Users/eshve/Downloads/chromedriver_win32/chromedriver.exe")
driver = webdriver.Chrome(executable_path="C:/Users/eshve/Downloads/chromedriver_win32/chromedriver.exe")
listofplayers = []

with open("C:/Users/eshve/Desktop/teams.csv",'r') as infile:
    for line in infile:
        filename = line.replace("\n","").split("-")
        outfile = open("C:/Users/eshve/Desktop/Teams/"+filename[1]+".csv",'w')
        driver.get(line)
        #teams = driver.find_elements_by_xpath('.//*[@class="team-link"]') 
        #teams = driver.find_elements_by_partial_link_text('/Teams/')
        players = driver.find_elements_by_xpath("//a[@href]")
        for player in players:
            text = player.get_attribute('href')
            if (text not in listofplayers):
                listofplayers.append(text)
                output = text + '\n'
                if ('/Players/' in output):
                    outfile.write(output)
                    print(output)
        outfile.close()
        time.sleep(2)
driver.close()