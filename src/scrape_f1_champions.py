import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException


def scrape_f1_champions():
    """
    Scrapes Formula 1 World Champions from StatsF1
    and saves the data to data/processed/f1_world_champions.csv
    """

    url = "https://www.statsf1.com/en/statistiques/pilote/champion/nombre.aspx"

    # ==================================
    # Project Paths
    # ==================================
    BASE_DIR = Path(__file__).resolve().parent.parent

    output_dir = BASE_DIR / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "f1_world_champions.csv"

    # ==================================
    # Selenium Setup
    # ==================================
    options = webdriver.ChromeOptions()

    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )

    driver = None

    try:
        print("🚀 Launching browser...")

        driver = webdriver.Chrome(options=options)
        driver.get(url)

        print("🌐 Opening StatsF1 website...")

        wait = WebDriverWait(driver, 15)

        import time

        time.sleep(10)

        print(driver.title)
        print(driver.current_url)

        with open("debug_page.html", "w", encoding="utf-8") as f:
            f.write(driver.page_source)

        print("Debug HTML saved")

        print("✅ Table loaded successfully!")

        page_source = driver.page_source

        # ==================================
        # Parse HTML
        # ==================================
        soup = BeautifulSoup(page_source, "html.parser")

        champions_table = soup.find(
            "table",
            id="ctl00_CPH_Main_GV_Stats"
        )

        if not champions_table:
            print("❌ Champions table not found.")
            return

        # ==================================
        # Extract Headers
        # ==================================
        header_row = champions_table.find("thead").find("tr")

        headers = [
            cell.get_text(strip=True)
            for cell in header_row.find_all(["td", "th"])
        ]

        print("Headers:", headers)

        # ==================================
        # Extract Rows
        # ==================================
        rows = []

        tbody = champions_table.find("tbody")

        for tr in tbody.find_all("tr"):
            cells = tr.find_all("td")

            row = []

            for cell in cells:
                row.append(
                    " ".join(cell.stripped_strings)
                )

            rows.append(row)

        # ==================================
        # Create DataFrame
        # ==================================
        df = pd.DataFrame(rows, columns=headers)

        # Remove useless Signature column
        if "Signature" in df.columns:
            df.drop(columns=["Signature"], inplace=True)

        # ==================================
        # Save CSV
        # ==================================
        df.to_csv(output_file, index=False)

        print(f"\n✅ Successfully scraped {len(df)} records")
        print(f"✅ File saved to:\n{output_file}")

        print("\n📊 Preview:")
        print(df.head())

    except TimeoutException:
        print("❌ Timed out while waiting for the table to load.")

    except Exception as e:
        print(f"❌ Error occurred: {e}")

    finally:
        if driver:
            driver.quit()
            print("\n🔒 Browser closed.")


if __name__ == "__main__":
    scrape_f1_champions()
