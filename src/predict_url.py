
import csv
from datetime import datetime
import pandas as pd
import joblib
import re
import ipaddress
import tldextract

from pathlib import Path
from urllib.parse import urlsplit

# =====================================
# 1. PROJECT PATHS
# =====================================

BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = BASE_DIR / "output"

OUTPUT_DIR.mkdir(exist_ok=True)

LOG_FILE = OUTPUT_DIR / "url_detection_logs.csv"

MODEL_FILE = BASE_DIR / "models" / "logistic_regression.pkl"

# =====================================
# 2. LOAD TRAINED MODEL
# =====================================

print("Loading Logistic Regression model...")

model = joblib.load(MODEL_FILE)

print("Model loaded successfully!")

# =====================================
# 3. URL FEATURE EXTRACTION
# =====================================

extractor = tldextract.TLDExtract(suffix_list_urls=())

SUSPICIOUS_WORDS = [
    "login", "verify", "update", "secure",
    "account", "banking", "password",
    "confirm", "wallet", "paypal",
    "signin", "free", "bonus"
]


def extract_features(url):

    url = url.strip()
    lower_url = url.lower()

    parsed = urlsplit(
        url if "://" in url else "//" + url
    )

    hostname = parsed.hostname or ""

    # 1. URL Length
    url_length = len(url)

    # 2. HTTPS
    has_https = int(
        url.lower().startswith("https://")
    )

    # 3. Dot Count
    dot_count = url.count(".")

    # 4. Hyphen Count
    hyphen_count = url.count("-")

    # 5. Special Characters
    special_chars = "@?=&%_+,$!#;~*"

    special_char_count = sum(
        url.count(c) for c in special_chars
    )

    # 6. Suspicious Words
    suspicious_word_count = sum(
        len(re.findall(re.escape(word), lower_url))
        for word in SUSPICIOUS_WORDS
    )

    # 7. IP Address
    try:
        ipaddress.ip_address(hostname)
        has_ip = 1
    except ValueError:
        has_ip = 0

    # 8. At Symbol
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
        c.isdigit() for c in url
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


def save_security_log(url, status, risk_score, risk_level, features):

    fieldnames = [
        "timestamp",
        "url",
        "status",
        "risk_score",
        "risk_level",
        "alert",
        "url_length",
        "has_https",
        "dot_count",
        "hyphen_count",
        "special_char_count",
        "suspicious_word_count",
        "has_ip",
        "has_at_symbol",
        "subdomain_count",
        "digit_count"
    ]

    alert = (
        "MALICIOUS_URL_DETECTED"
        if status == "MALICIOUS"
        else "NONE"
    )

    log_record = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "url": url,
        "status": status,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "alert": alert,
        **features
    }

    file_exists = LOG_FILE.exists()

    with open(LOG_FILE, "a", newline="", encoding="utf-8") as file:

        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists or LOG_FILE.stat().st_size == 0:
            writer.writeheader()

        writer.writerow(log_record)

    print("\nSecurity log saved successfully!")

    if status == "MALICIOUS":
        print("!!! SECURITY ALERT: MALICIOUS URL DETECTED !!!")

# =====================================
# 4. URL PREDICTION
# =====================================

def predict_url(url):

    features = extract_features(url)

    # Convert features into DataFrame
    input_data = pd.DataFrame([features])

    # Predict class
    prediction = int(model.predict(input_data)[0])

    # Get malicious probability
    probability = model.predict_proba(input_data)[0][1]

    # Risk score
    risk_score = round(probability * 100, 2)

    # Classification
    if prediction == 1:
        status = "MALICIOUS"
    else:
        status = "LEGITIMATE"

    # Risk level
    if risk_score >= 70:
        risk_level = "HIGH"
    elif risk_score >= 30:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    save_security_log(
    url,
    status,
    risk_score,
    risk_level,
    features
)    

    # Display result
    print("\n========== URL ANALYSIS ==========")

    print("URL:", url)
    print("Status:", status)
    print("Risk Score:", risk_score, "/100")
    print("Risk Level:", risk_level)

    print("\nExtracted Features:")

    for name, value in features.items():
        print(name, ":", value)

    print("\n==================================")


# =====================================
# 5. MAIN PROGRAM
# =====================================

print("\nMalicious URL Detection System")

while True:

    url = input(
        "\nEnter URL (or type exit): "
    ).strip()

    if url.lower() == "exit":
        print("Exiting URL Detection System.")
        break

    if not url:
        print("Please enter a URL.")
        continue

    try:
        predict_url(url)

    except Exception as e:
        print("Prediction error:", e)
