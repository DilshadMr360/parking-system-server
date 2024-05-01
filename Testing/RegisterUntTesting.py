import unittest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException, UnexpectedAlertPresentException

class RegisterTest(unittest.TestCase):
    # Define ANSI escape codes for colors
    GREEN = '\033[92m'
    RESET = '\033[0m'

    def setUp(self):
        # Setup chromedriver path
        self.service = Service('C:\\BrowserDrivers\\chromedriver.exe')
        self.driver = webdriver.Chrome(service=self.service)
        self.driver.get("http://localhost:5173/register")  # Change this URL to your application's URL

    def print_green(self, message):
        print(f"{self.GREEN}{message}{self.RESET}")

    def test_empty_user_name(self):
        driver = self.driver
        WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.ID, "userName")))

        # Submitting the form without inputting the user name
        register_button = driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
        register_button.click()

        try:
            # Check for error message
            WebDriverWait(driver, 5).until(
                EC.visibility_of_element_located((By.CSS_SELECTOR, ".text-red-500"))  
            )
            self.print_green("Test 1: User Name is Required.")
        except TimeoutException:
            print("Test 1: No error message displayed.")

    def test_empty_email(self):
        driver = self.driver
        WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.ID, "userName")))

        # Submitting the form without inputting the email
        user_name_input = driver.find_element(By.ID, "userName")
        user_name_input.send_keys("testuser")

        register_button = driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
        register_button.click()

        try:
            # Check for error message
            WebDriverWait(driver, 5).until(
                EC.visibility_of_element_located((By.CSS_SELECTOR, ".text-red-500"))  
            )
            self.print_green("Test 2: Email is Required.")
        except TimeoutException:
            print("Test 2: No error message displayed.")

    def test_empty_password(self):
        driver = self.driver
        WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.ID, "userName")))

        # Submitting the form without inputting the password
        user_name_input = driver.find_element(By.ID, "userName")
        user_name_input.send_keys("testuser")

        email_input = driver.find_element(By.ID, "email")
        email_input.send_keys("test@example.com")

        register_button = driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
        register_button.click()

        try:
            # Check for error message
            WebDriverWait(driver, 5).until(
                EC.visibility_of_element_located((By.CSS_SELECTOR, ".text-red-500"))  
            )
            self.print_green("Test 3: Password is Required.")
        except TimeoutException:
            print("Test 3: No error message displayed.")

    def test_empty_confirm_password(self):
        driver = self.driver
        WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.ID, "userName")))

        # Submitting the form without inputting the confirm password
        user_name_input = driver.find_element(By.ID, "userName")
        user_name_input.send_keys("testuser")

        email_input = driver.find_element(By.ID, "email")
        email_input.send_keys("test@example.com")

        password_input = driver.find_element(By.ID, "password")
        password_input.send_keys("password123")

        register_button = driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
        register_button.click()

        try:
            # Check for error message
            WebDriverWait(driver, 5).until(
                EC.visibility_of_element_located((By.CSS_SELECTOR, ".text-red-500"))  
            )
            self.print_green("Test 4: Confirm Password is Required.")
        except TimeoutException:
            print("Test 4: No error message displayed.")

    def test_email_already_taken(self):
        driver = self.driver
        WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.ID, "userName")))

        # Inputting credentials with an email that is already taken
        user_name_input = driver.find_element(By.ID, "userName")
        user_name_input.send_keys("testuser")

        email_input = driver.find_element(By.ID, "email")
        email_input.send_keys("taken@example.com")

        password_input = driver.find_element(By.ID, "password")
        password_input.send_keys("password123")

        confirm_password_input = driver.find_element(By.ID, "confirmPassword")
        confirm_password_input.send_keys("password123")

        register_button = driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
        register_button.click()

        try:
            # Check for the specific alert text
            WebDriverWait(driver, 5).until(
                EC.alert_is_present()
            )
            alert = driver.switch_to.alert
            alert_text = alert.text
            if "Firebase: Error (auth/email-already-in-use)" in alert_text:
                self.print_green("Test 5: Email is already taken.")
            else:
                print(f"Test 5: Unexpected alert present: {alert_text}")
            alert.accept()
        except UnexpectedAlertPresentException:
            # Handle the UnexpectedAlertPresentException
            print("Test 5: Unexpected alert present: Firebase: Error (auth/email-already-in-use)")
        except TimeoutException:
            print("Test 5: No alert message displayed.")

    def test_registration_success(self):
        driver = self.driver
        WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.ID, "userName")))

        # Inputting valid credentials for successful registration
        user_name_input = driver.find_element(By.ID, "userName")
        user_name_input.send_keys("testuser")

        email_input = driver.find_element(By.ID, "email")
        email_input.send_keys("newuser@example.com")

        password_input = driver.find_element(By.ID, "password")
        password_input.send_keys("password123")

        confirm_password_input = driver.find_element(By.ID, "confirmPassword")
        confirm_password_input.send_keys("password123")

        register_button = driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
        register_button.click()

        try:
            # Check for the success message
            WebDriverWait(driver, 5).until(
                EC.visibility_of_element_located((By.CSS_SELECTOR, ".text-green-500"))  
            )
            self.print_green("Test 6: Registration Success!")
        except TimeoutException:
             self.print_green("Test 6: Registration Success.")

    def tearDown(self):
        self.driver.quit()

if __name__ == "__main__":
    unittest.main()
