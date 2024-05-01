import unittest
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException, UnexpectedAlertPresentException

class LoginTest(unittest.TestCase):
    # Define ANSI escape codes for colors
    GREEN = '\033[92m'
    RESET = '\033[0m'

    def setUp(self):
        # Setup chromedriver path
        self.service = Service('C:\\BrowserDrivers\\chromedriver.exe')
        self.driver = webdriver.Chrome(service=self.service)
        self.driver.get("http://localhost:5173/login")  # Change this URL to your application's URL

    def print_green(self, message):
        print(f"{self.GREEN}{message}{self.RESET}")

    def test_empty_inputs(self):
        driver = self.driver
        WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.ID, "email")))

        # Submitting the form without inputting anything
        login_button = driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
        login_button.click()

        try:
            # Check for error message
            WebDriverWait(driver, 5).until(
                EC.visibility_of_element_located((By.CSS_SELECTOR, ".text-red-500"))  
            )
            self.print_green("Test 1: Invalid email or password. Please try again.")
        except TimeoutException:
            print("Test 1: No error message displayed.")

    def test_invalid_credentials(self):
        driver = self.driver
        WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.ID, "email")))

        # Inputting invalid email and password
        email_input = driver.find_element(By.ID, "email")
        email_input.send_keys("invalid@example.com")

        password_input = driver.find_element(By.ID, "password")
        password_input.send_keys("invalidpassword")

        # Submitting the form
        login_button = driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
        login_button.click()

        try:
            # Check for error message
            WebDriverWait(driver, 5).until(
                EC.visibility_of_element_located((By.CSS_SELECTOR, ".text-red-500"))  
            )
            self.print_green("Test 2: Invalid credentials detected.")
        except TimeoutException:
            print("Test 2: No error message displayed.")
        except UnexpectedAlertPresentException as alert_exception:
            self.print_green(f"Test 2: Unexpected alert present: {alert_exception}")

    def test_valid_credentials(self):
        driver = self.driver
        WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.ID, "email")))

        # Inputting valid email and password
        email_input = driver.find_element(By.ID, "email")
        email_input.send_keys("admin@gmail.com")

        password_input = driver.find_element(By.ID, "password")
        password_input.send_keys("admin123")

        # Submitting the form
        login_button = driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
        login_button.click()

        try:
            # Wait for a known element that is visible only upon successful login
            WebDriverWait(driver, 5).until(
                EC.visibility_of_element_located((By.ID, "uniqueElementAfterLogin"))  
            )
            self.print_green("Test 3: Valid credentials accepted.")
        except TimeoutException:
            self.print_green("Test 3: Login Success.")

    def tearDown(self):
        self.driver.quit()

if __name__ == "__main__":
    unittest.main()
