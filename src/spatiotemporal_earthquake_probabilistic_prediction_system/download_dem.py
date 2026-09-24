import sys
import urllib.request
import zipfile
from pathlib import Path

DEM_URL = "https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/topo/downloads/GMTED/Grid_ZipFiles/mi30_grd.zip"

# Pin paths relative to this script's directory
MODULE_DIR = Path(__file__).resolve().parent
EXTRACT_DIR = MODULE_DIR / "gmted2010"
ZIP_PATH = MODULE_DIR / "mi30_grd.zip"
CHECK_FILE = EXTRACT_DIR / "mi30_grd" / "hdr.adf"


def _download_with_retries(url: str, dest: Path, max_retries: int = 3):
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    for attempt in range(1, max_retries + 1):
        try:
            print(f"Connecting to USGS server (Attempt {attempt}/{max_retries})...")
            req = urllib.request.Request(url, headers=headers)

            with urllib.request.urlopen(req, timeout=60) as response, open(dest, "wb") as out_file:
                total_size = int(response.headers.get("Content-Length", 0))
                downloaded = 0
                block_size = 128 * 1024  # 128 KB chunk size

                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    downloaded += len(buffer)
                    out_file.write(buffer)

                    if total_size > 0:
                        percent = min(100, int(downloaded * 100 / total_size))
                        mb_dl = downloaded / (1024 * 1024)
                        mb_tot = total_size / (1024 * 1024)
                        sys.stdout.write(
                            f"\rDownloading DEM dataset: {percent}% ({mb_dl:.1f}/{mb_tot:.1f} MB)"
                        )
                        sys.stdout.flush()

                sys.stdout.write("\n")

            if total_size > 0 and downloaded < total_size:
                raise IOError(f"Incomplete download: got {downloaded} of {total_size} bytes.")

            return  # Success

        except Exception as e:
            print(f"\nDownload attempt {attempt} failed: {e}")
            if dest.exists():
                dest.unlink()
            if attempt == max_retries:
                raise


def ensure_dem_dataset():
    if CHECK_FILE.exists():
        print(f"DEM dataset already exists at '{CHECK_FILE}'. Skipping download.")
        return

    print("DEM dataset not found. Fetching from USGS archive...")

    try:
        _download_with_retries(DEM_URL, ZIP_PATH)
        print("Download complete. Extracting archive...")

        EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)

        print(f"Extraction complete. Target location: {CHECK_FILE}")

    finally:
        if ZIP_PATH.exists():
            ZIP_PATH.unlink()
            print("Cleaned up temporary zip file.")


if __name__ == "__main__":
    ensure_dem_dataset()