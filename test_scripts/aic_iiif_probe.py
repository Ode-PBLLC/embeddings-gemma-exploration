"""Probe whether a headless browser gets past the Cloudflare challenge on AIC IIIF."""
import sys
from playwright.sync_api import sync_playwright

ID = "fec5d023-0b46-45df-eb57-cbf64e24121d"
URL = f"https://www.artic.edu/iiif/2/{ID}/full/843,/0/default.jpg"
headless = "--headed" not in sys.argv
with sync_playwright() as p:
    b = p.chromium.launch(headless=headless)
    ctx = b.new_context()
    page = ctx.new_page()
    r = page.goto(URL, timeout=60000)
    print("goto", r.status, r.headers.get("content-type"))
    page.wait_for_timeout(8000)
    r2 = ctx.request.get(URL)
    print("ctx.request", r2.status, r2.headers.get("content-type"), len(r2.body()))
    print("cookies", [c["name"] for c in ctx.cookies()])
    print("ua", page.evaluate("navigator.userAgent"))
    b.close()
