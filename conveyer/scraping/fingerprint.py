"""Hosting-platform fingerprints from response headers and markup.

"Anything can be useful for the model": even when a site bot-walls the body,
the **response headers** usually still identify the software serving it —
``x-shopid`` is Shopify, ``x-wix-request-id`` is Wix, ``x-magento-*`` is
Magento. A commerce platform on a domain no curated list knows is storefront
evidence: it corroborates a ``/products/…`` URL that could not be fetched and
supplies a ``seller_type`` hint (self-hosted store platforms are
overwhelmingly brand-owned DTC shops; marketplaces and retailers run their
own stacks).

The detector is deliberately conservative: header signatures first (they
survive blocks), then unambiguous markup strings (``cdn.shopify.com``,
``wp-content/plugins/woocommerce``). CDN/WAF vendors (Cloudflare, Akamai,
Fastly) are recorded as infrastructure — never as commerce evidence, since a
WAF block page says nothing about what it protects.
"""

from __future__ import annotations

from typing import Dict, Optional

# platform -> is it a commerce storefront platform?
COMMERCE_PLATFORMS = {
    "shopify", "woocommerce", "bigcommerce", "magento", "salesforce_commerce",
    "prestashop", "shopware", "squarespace_commerce", "wix_stores",
}
INFRA_PLATFORMS = {"cloudflare", "akamai", "fastly", "vercel", "netlify",
                   "wordpress", "wix", "squarespace", "webflow", "drupal"}

# header-name (lowercase, prefix match) -> platform. Ordered: most specific
# first, commerce before infra so "shopify behind cloudflare" reads shopify.
_HEADER_SIGNS = [
    ("x-shopify-stage", "shopify"), ("x-shopid", "shopify"),
    ("x-sorting-hat-shopid", "shopify"), ("x-shardid", "shopify"),
    ("x-storefront-renderer", "shopify"),
    ("x-bc-", "bigcommerce"),
    ("x-magento-", "magento"),
    ("x-dw-request-base-id", "salesforce_commerce"),
    ("x-sf-cc-", "salesforce_commerce"),
    ("x-prestashop-", "prestashop"),
    ("x-shopware-", "shopware"),
    ("x-wix-request-id", "wix"), ("x-wix-", "wix"),
    ("x-contextid", "squarespace"),
    ("x-served-by", None),           # varnish/fastly chatter — ignore, too generic
    ("cf-ray", "cloudflare"), ("cf-cache-status", "cloudflare"),
    ("x-akamai-", "akamai"), ("akamai-", "akamai"),
    ("x-vercel-", "vercel"), ("x-nf-request-id", "netlify"),
]
# (server-header substring, platform)
_SERVER_SIGNS = [
    ("cloudflare", "cloudflare"), ("akamai", "akamai"),
    ("bigcommerce", "bigcommerce"), ("squarespace", "squarespace"),
]
# unambiguous markup substrings (lowercase) -> platform
_HTML_SIGNS = [
    ("cdn.shopify.com", "shopify"), ("shopify.theme", "shopify"),
    ("wp-content/plugins/woocommerce", "woocommerce"),
    ('content="woocommerce', "woocommerce"),
    ("mage/cookies", "magento"), ("magento_", "magento"),
    ("demandware.static", "salesforce_commerce"),
    ("cdn11.bigcommerce.com", "bigcommerce"),
    ("static.parastorage.com", "wix"),
    ("static1.squarespace.com", "squarespace"),
    ("prestashop", "prestashop"),
    ('content="wordpress', "wordpress"),
    ("assets.website-files.com", "webflow"),
]


def detect_platform(headers: Optional[Dict[str, str]], html: str = "") -> str:
    """Best-effort platform name ('' when nothing matches). Commerce platforms
    win over infrastructure: a Shopify store behind Cloudflare is 'shopify'."""
    found_infra = ""
    hdrs = {str(k).lower(): str(v).lower() for k, v in (headers or {}).items()}
    for sign, platform in _HEADER_SIGNS:
        if platform is None:
            continue
        if any(k.startswith(sign) for k in hdrs):
            if platform in INFRA_PLATFORMS:
                found_infra = found_infra or platform
            else:
                return platform
    server = hdrs.get("server", "")
    powered = hdrs.get("x-powered-by", "")
    for sign, platform in _SERVER_SIGNS:
        if sign in server or sign in powered:
            if platform in INFRA_PLATFORMS:
                found_infra = found_infra or platform
            else:
                return platform
    blob = (html or "")[:200_000].lower()
    if blob:
        for sign, platform in _HTML_SIGNS:
            if sign in blob:
                if platform == "wordpress" and "woocommerce" in blob:
                    return "woocommerce"
                if platform in INFRA_PLATFORMS:
                    found_infra = found_infra or platform
                else:
                    return platform
        # squarespace/wix storefront variants: the builder plus commerce markup
        if "static1.squarespace.com" in blob and "sqs-add-to-cart" in blob:
            return "squarespace_commerce"
    return found_infra


def is_commerce_platform(platform: str) -> bool:
    return platform in COMMERCE_PLATFORMS


def platform_seller(platform: str) -> str:
    """seller_type hint: self-hosted store platforms ⇒ brand-owned DTC.
    Marketplaces/retailers run custom stacks, so this never fires for them."""
    return "brand_owned" if platform in COMMERCE_PLATFORMS else ""
