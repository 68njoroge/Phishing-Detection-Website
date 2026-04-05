import re
from urllib.parse import urlparse
import tldextract
import numpy as np
import string

SUSPICIOUS_TERMS = [
    'login', 'secure', 'bank', 'account', 'update', 'verify',
    'paypal', 'amazon', 'ebay', 'roblox', 'confirm', 'signin',
    'webscr', 'wp-admin', 'admin', 'checkout', 'support',
    'password', 'credential', 'wallet', 'billing', 'invoice',
    'suspended', 'unusual', 'activity', 'click', 'free', 'prize',
    'winner', 'urgent', 'alert', 'access', 'validate', 'reset'
]

BRAND_TERMS = [
    'paypal', 'amazon', 'apple', 'google', 'microsoft', 'netflix',
    'facebook', 'instagram', 'twitter', 'ebay', 'walmart', 'chase',
    'wellsfargo', 'bankofamerica', 'steam', 'roblox', 'discord'
]

DEFAULT_CHARS = string.ascii_lowercase + string.digits + "-._~:/?#[]@!$&'()*+,;=%+ "


def _ensure_scheme(url: str) -> str:
    url = url.strip()
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "http://" + url
    return url


def is_brand_spoofed(netloc: str) -> int:
    extracted   = tldextract.extract(netloc)
    root_domain = extracted.domain.lower()

    for brand in BRAND_TERMS:
        brand_core = brand.replace('.com', '').lower()
        if brand_core in netloc.lower():
            if brand_core != root_domain:
                return 1
    return 0


def extract_lexical_features(url):
    url = _ensure_scheme(url)

    parsed_url = urlparse(url)
    netloc = parsed_url.netloc
    path   = parsed_url.path
    query  = parsed_url.query

    features = {}

    features['url_length']    = len(url)
    features['domain_length'] = len(netloc)
    features['path_length']   = len(path)
    features['query_length']  = len(query)

    features['num_dots']          = url.count('.')
    features['num_hyphens']       = url.count('-')
    features['num_digits']        = sum(c.isdigit() for c in url)
    features['num_special_chars'] = sum(
        not c.isalnum() for c in url if c not in ('/', '.', ':', '-')
    )
    features['num_subdomains'] = netloc.count('.') - 1 if netloc.count('.') > 1 else 0

    features['suspicious_term_count'] = sum(
        1 for term in SUSPICIOUS_TERMS if term in url.lower()
    )
    features['has_suspicious_terms'] = (
        1 if features['suspicious_term_count'] > 0 else 0
    )

    features['brand_spoofed']     = is_brand_spoofed(netloc)
    features['brand_in_domain']   = sum(1 for b in BRAND_TERMS if b in netloc.lower())
    features['hyphens_in_domain'] = netloc.count('-')
    features['long_domain']       = 1 if len(netloc) > 30 else 0

    features['is_https'] = 1 if parsed_url.scheme == 'https' else 0

    features['has_ip'] = (
        1 if re.match(r'^\d{1,3}(\.\d{1,3}){3}$', netloc) else 0
    )

    features['digit_ratio']  = features['num_digits']  / max(len(url), 1)
    features['dot_ratio']    = features['num_dots']    / max(len(url), 1)
    features['hyphen_ratio'] = features['num_hyphens'] / max(len(netloc), 1)

    features['has_at_symbol']    = 1 if '@' in url else 0
    features['has_double_slash'] = 1 if '//' in path else 0
    features['has_punycode']     = 1 if 'xn--' in url.lower() else 0

    return features


def extract_host_features(url):
    url = _ensure_scheme(url)

    parsed = tldextract.extract(url)
    features = {}

    features['domain_is_ip'] = (
        1 if re.match(r'^\d{1,3}(\.\d{1,3}){3}$', parsed.domain) else 0
    )

    features['has_subdomain']    = 1 if parsed.subdomain else 0
    features['tld_length']       = len(parsed.suffix)
    features['subdomain_levels'] = (
        parsed.subdomain.count('.') + 1 if parsed.subdomain else 0
    )

    suspicious_tlds = {
        'tk', 'ml', 'ga', 'cf', 'gq', 'xyz', 'top', 'club',
        'online', 'site', 'info', 'biz', 'pw', 'cc'
    }
    features['suspicious_tld']   = 1 if parsed.suffix.lower() in suspicious_tlds else 0
    features['digits_in_domain'] = sum(c.isdigit() for c in parsed.domain)

    return features


def extract_all_features(url):
    lexical = extract_lexical_features(url)
    host    = extract_host_features(url)
    return {**lexical, **host}


def url_to_sequence(url, max_len=200, char_map=None):
    url = url.lower().strip()
    url = re.sub(r'https?://', '', url)
    url = re.sub(r'^www\.', '', url)

    if char_map is None:
        char_map = {c: i + 1 for i, c in enumerate(DEFAULT_CHARS)}

    seq  = [char_map.get(c, 0) for c in url[:max_len]]
    seq += [0] * (max_len - len(seq))
    return np.array(seq)