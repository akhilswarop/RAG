from linkedin_scraper import JobSearch, actions
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# Configure Chrome WebDriver options
chrome_options = Options()
chrome_options.add_argument("--disable-gpu")  # Fix for GPU-related errors
chrome_options.add_argument("--disable-software-rasterizer")  # Ensure software rendering
chrome_options.add_argument("--disable-dev-shm-usage")  # Improve resource management
chrome_options.add_argument("--no-sandbox")  # For running in containers
# Uncomment the next line if you want the browser to run in the background (headless mode)
# chrome_options.add_argument("--headless")

# Initialize WebDriver
driver = webdriver.Chrome(
    service=Service(ChromeDriverManager().install()),
    options=chrome_options
)

# LinkedIn credentials
email = "akhiltheswarop@gmail.com"
password = "Vaazhkai@12"

# Log in to LinkedIn
actions.login(driver, email, password)  # Prompts in terminal if email/password aren't provided
input("Press Enter after successful login in the browser window.")

# Correct instantiation of JobSearch
job = JobSearch(
    driver=driver,
    base_url="https://www.linkedin.com/jobs/collections/recommended/?currentJobId=3456898261",
    close_on_complete=False
)

# Perform job scraping logic (if needed, you can expand this)
print("Job search object created successfully.")

# Quit the driver at the end to clean up resources
driver.quit()
