import re
import ipaddress
from urllib.parse import urlparse

SUSPICIOUS_KEYWORDS = [
    "login", "verify", "secure", "account", "update",
    "confirm", "password", "banking", "signin",
]

SHORTENER_DOMAINS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly",
    "adf.ly", "tiny.cc", "rebrand.ly", "cutt.ly", "shorte.st", "rb.gy",
    "s.id", "v.gd", "shorturl.at", "bl.ink", "lnkd.in", "soo.gd", "clck.ru",
}

MULTI_PART_SUFFIXES = {
    "co.uk", "org.uk", "ac.uk", "gov.uk", "me.uk", "net.uk", "sch.uk",
    "co.jp", "ne.jp", "or.jp", "ac.jp", "co.kr", "co.in", "org.in", "net.in",
    "co.nz", "org.nz", "co.za", "org.za", "com.au", "net.au", "org.au",
    "com.br", "com.cn", "net.cn", "org.cn", "com.mx", "com.tr", "com.sg",
    "co.id", "or.id", "com.hk", "org.hk", "com.tw", "co.th", "com.vn",
    "co.il", "com.ar", "com.co",
}


def get_hostname(url: str) -> str:
    try:
        parsed = urlparse(url if "//" in url else "//" + url)
        host = parsed.hostname or ""
        return host.lower()
    except Exception:
        return ""


def is_ip_address(hostname: str) -> bool:
    try:
        ipaddress.ip_address(hostname)
        return True
    except ValueError:
        return False


def split_domain(hostname: str):
    if not hostname or is_ip_address(hostname):
        return [], hostname, ""
    labels = hostname.split(".")
    if len(labels) <= 2:
        return [], hostname, (labels[-1] if labels else "")
    last_two = ".".join(labels[-2:])
    if last_two in MULTI_PART_SUFFIXES and len(labels) > 2:
        suffix = last_two
        registrable = ".".join(labels[-3:])
        subdomains = labels[:-3]
    else:
        suffix = labels[-1]
        registrable = ".".join(labels[-2:])
        subdomains = labels[:-2]
    return subdomains, registrable, suffix


def score_url_length(url: str) -> int:
    n = len(url)
    if n > 120:
        return 2
    if n >= 75:
        return 1
    return 0


def score_ip_address(hostname: str) -> int:
    return 3 if is_ip_address(hostname) else 0


def score_subdomain(hostname: str) -> int:
    subs, _, _ = split_domain(hostname)
    k = len(subs)
    if k >= 3:
        return 2
    if k == 2:
        return 1
    return 0


def score_keywords(url: str) -> int:
    low = url.lower()
    hits = sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in low)
    return min(hits, 2)


def score_at_symbol(url: str) -> int:
    return 2 if "@" in url else 0


def score_special_patterns(url: str) -> int:
    count = 0
    if ".." in url:
        count += 1
    if url.count("-") >= 4:
        count += 1
    if re.search(r"%[0-9A-Fa-f]{2}", url):
        count += 1
    after_scheme = re.sub(r"^[a-zA-Z]+://", "", url)
    if "//" in after_scheme:
        count += 1
    return min(count, 2)


def score_https(url: str) -> int:
    return 0 if url.lower().startswith("https://") else 1


def score_shortener(hostname: str) -> int:
    return 2 if hostname in SHORTENER_DOMAINS else 0


def score_punycode(hostname: str) -> int:
    return 3 if "xn--" in hostname else 0


def score_digit_hyphen_ratio(url: str, digit_thresh: float = 0.15, hyphen_thresh: float = 0.10) -> int:
    n = len(url) or 1
    digit_ratio = sum(c.isdigit() for c in url) / n
    hyphen_ratio = url.count("-") / n
    return 1 if (digit_ratio > digit_thresh or hyphen_ratio > hyphen_thresh) else 0


FEATURE_NAMES = [
    "ip_address", "punycode", "at_symbol", "shortener",
    "url_length", "subdomain", "keyword", "special_pattern",
    "https_absent", "digit_hyphen_ratio",
]

DEFAULT_WEIGHTS = {
    "ip_address": 3,
    "punycode": 3,
    "at_symbol": 2,
    "shortener": 2,
    "url_length": 2,
    "subdomain": 2,
    "keyword": 1,
    "special_pattern": 1,
    "https_absent": 1,
    "digit_hyphen_ratio": 1,
}

BINARY_FEATURES = {"ip_address", "punycode", "at_symbol", "shortener", "https_absent", "digit_hyphen_ratio"}
BANDED_FEATURES = {"url_length", "subdomain", "keyword", "special_pattern"}


def extract_raw_subscores(url: str) -> dict:
    hostname = get_hostname(url)
    return {
        "ip_address": 1 if is_ip_address(hostname) else 0,
        "punycode": 1 if "xn--" in hostname else 0,
        "at_symbol": 1 if "@" in url else 0,
        "shortener": 1 if hostname in SHORTENER_DOMAINS else 0,
        "url_length": score_url_length(url),
        "subdomain": score_subdomain(hostname),
        "keyword": score_keywords(url),
        "special_pattern": score_special_patterns(url),
        "https_absent": score_https(url),
        "digit_hyphen_ratio": score_digit_hyphen_ratio(url),
    }


def weighted_score(subscores: dict, weights: dict) -> float:
    total = 0.0
    for name, sub in subscores.items():
        w = weights.get(name, 0)
        if name in BANDED_FEATURES:
            total += w * (sub / 2.0)
        else:
            total += w * sub
    return total


def risk_category(score: float, low_high: int, med_high: int) -> str:
    if score <= low_high:
        return "low"
    if score <= med_high:
        return "medium"
    return "high"
