
import pandas as pd
import re
import ipaddress
import tldextract

from pathlib import Path
from urllib.parse import urlsplit

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "dataset" / "processed" / "cleaned_urls.csv"
OUTPUT_DIR = BASE_DIR / "dataset" / "processed"

OUTPUT_FILE = OUTPUT_DIR / "features_dataset.csv"

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Avoid downloading suffix data from the internet
extractor = tldextract.TLDExtract(suffix_list_urls=())

# Suspicious keywords
SUSPICIOUS_WORDS = [
    "login", "verify", "update", "secure",
    "account", "banking", "password",
    "confirm", "wallet", "paypal",
    "signin", "free", "bonus"
]


def extract_features(url):

    url = str(url).strip()
    lower_url = url.lower()

    # Parse URL safely
    parsed = urlsplit(
        url if "://" in url else "//" + url
    )

    hostname = parsed.hostname or ""

    # 1. URL Length
    url_length = len(url)

    # 2. HTTPS Detection
    has_https = int(
        url.lower().startswith("https://")
    )

    # 3. Dot Count
    dot_count = url.count(".")

    # 4. Hyphen Count
    hyphen_count = url.count("-")

    # 5. Special Character Count
    special_chars = "@?=&%_+,$!#;~*"

    special_char_count = sum(
        url.count(char) for char in special_chars
    )

    # 6. Suspicious Keyword Count
    suspicious_word_count = sum(
        len(re.findall(
            re.escape(word),
            lower_url
        ))
        for word in SUSPICIOUS_WORDS
    )

    # 7. IP Address Detection
    try:
        ipaddress.ip_address(hostname)
        has_ip = 1
    except ValueError:
        has_ip = 0

    # 8. At Symbol Detection
    has_at_symbol = int("@" in url)

    # 9. Subdomain Count
    try:
        extracted = extractor(hostname)

        if extracted.domain and extracted.suffix:
            subdomain_count = (
                len(extracted.subdomain.split("."))
                if extracted.subdomain else 0
            )
        else:
            subdomain_count = 0

    except Exception:
        subdomain_count = 0

    # 10. Digit Count
    digit_count = sum(
        char.isdigit() for char in url
    )

    return {
        "url_length": url_length,
        "has_https": has_https,
        "dot_count": dot_count,
        "hyphen_count": hyphen_count,
        "special_char_count": special_char_count,
        "suspicious_word_count": suspicious_word_count,
        "has_ip": has_ip,
        "has_at_symbol": has_at_symbol,
        "subdomain_count": subdomain_count,
        "digit_count": digit_count
    }


# Main program
print("Loading cleaned URL dataset...")

df = pd.read_csv(INPUT_FILE)

print("Total URLs:", len(df))

print("\nExtracting URL features...")

# Extract features
feature_data = df["url"].apply(extract_features)

features_df = pd.DataFrame(
    feature_data.tolist()
)

# Combine original URL, features and label
final_df = pd.concat(
    [
        df[["url"]],
        features_df,
        df[["label"]]
    ],
    axis=1
)

# Save feature dataset
final_df.to_csv(OUTPUT_FILE, index=False)

# Display results
print("\n========== FEATURE EXTRACTION RESULTS ==========")

print("Total records:", len(final_df))

print("\nExtracted Features:")
print(features_df.columns.tolist())

print("\nSample Feature Records:")
print(final_df.head(10).to_string(index=False))

print("\nFeature Dataset Shape:", final_df.shape)

print("\nFeature extraction completed successfully!")

print("Saved file:", OUTPUT_FILE)
