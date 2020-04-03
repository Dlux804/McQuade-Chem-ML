import pickle
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver import ActionChains
import pyautogui
import time
import timeit
import requests

VCU_username = "hyerap"
VCU_password = "Runningwithoutpants##01"

SciFinder_username = "andres_hyer"
SciFinder_password = "Cowboys##01"

driver = webdriver.Chrome()
driver.fullscreen_window()
driver.get('https://scifinder-cas-org.proxy.library.vcu.edu/scifinder/view/scifinder/scifinderExplore.jsf')

driver.find_element_by_css_selector("#username").send_keys(VCU_username)
driver.find_element_by_css_selector("#password").send_keys(VCU_password)
driver.find_element_by_css_selector("#fm1 > div:nth-child(3) > button").click()

WebDriverWait(driver, 60).until(
    EC.presence_of_element_located((By.CSS_SELECTOR, "#username"))
)

driver.find_element_by_css_selector("#username").send_keys(SciFinder_username)
driver.find_element_by_css_selector("#password").send_keys(SciFinder_password)
driver.find_element_by_css_selector("#loginButton").click()

time.sleep(60)
pyautogui.moveTo(309, 213)
pyautogui.hotkey('ctrl', 'shift', 'i')
time.sleep(5)
pyautogui.rightClick()
pyautogui.press(['up', 'enter'])
pyautogui.moveTo(1349, 339)
pyautogui.rightClick()
pyautogui.press(['right', 'down', 'down', 'down', 'right',
                 'up', 'up', 'up', 'up', 'up', 'enter'])

timer = timeit.default_timer()
while True:
    print(pyautogui.position())
    if timeit.default_timer()-timer > 120:
        break
    time.sleep(0.01)

'''
actionChains = ActionChains(driver)
actionChains.move_to_element_with_offset(searchbar, 100, 100)
actionChains.context_click()
actionChains.perform()  # Context_click is right click

driver.get('https://scifinder-cas-org.proxy.library.vcu.edu/scifinder/view/scifinder/scifinderExplore.jsf')
cookies = pickle.load(open("cookies.pkl", "rb"))
for cookie in cookies:
    driver.add_cookie(cookie)
input('Press enter after logging into SciFinder to begin scraping \n')
pickle.dump(driver.get_cookies(), open("cookies.pkl","wb"))


i = 1
while i <= 50:
    css = "#listContent > ol > li:nth-child({}) > div".format(str(i))
    print(css)
    reaction_text_s = driver.find_elements_by_css_selector(css)
    print(reaction_text_s)
    for reaction_text in reaction_text_s:
        with open("SciFinder_Scraping_Data/{}.txt".format(str(i)), "w+") as file:
            file.write(reaction_text)
    i = i + 1
'''

# driver.close()
