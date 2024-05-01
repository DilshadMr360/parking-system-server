from selenium import webdriver
from selenium.webdriver.chrome.service import Service
import time

# Specify the path to chromedriver using the Service class
service = Service(executable_path="C:\\BrowserDrivers\\chromedriver.exe")

# Initialize Chrome WebDriver with the specified service
driver = webdriver.Chrome(service=service)

# Open the login page
driver.get("http://localhost:5173/login")

# Maximize the window to ensure the title is visible
driver.maximize_window()

# Print the title of the page
print(driver.title)

# Wait for 5 seconds (adjust as needed)
time.sleep(5)

# Close the browser window
driver.quit()
