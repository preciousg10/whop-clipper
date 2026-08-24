"""hunt.py — TIERED footage resolution for intake.

Scout hands over links; intake must resolve WHERE the real footage actually is before giving
up. Two tiers, simplest-first — only escalate to a browser when truly needed:

  TIER 1 (NO browser): direct footage (YouTube/Drive/Kick/direct video URL), Google Docs
  (fetch text + follow the footage links inside), Drive folders (list/route their contents),
  following freely-reachable links up to MAX_HOPS hops (resource → doc → drive/link). Handled
  by intake's frontier loop using the download machinery + these helpers.

  TIER 2 (Playwright, ONLY when needed): a THIRD-PARTY website whose footage links a simple
  fetch can't extract (JS-rendered) → load it in a headless browser and scrape the footage
  links off the page. NEVER used for Drive/YouTube/Doc — Tier 1 handles those faster.

HARD STOP: if reaching footage requires login / signup / payment / any manual human action
(email-to-request, DM, application) we NEVER proceed — the campaign is skipped (intake fails
loud → the auto-advance walk moves on). We never enter credentials or pay, ever.
`detect_barrier()` flags such a gate; intake only ACTS on it when NO freely-reachable footage
was found (a gate we can route around is ignored — the barrier must stand BETWEEN us and the
footage).

Playwright is an OPTIONAL dependency (like the TRACK-mode reframe deps): if it (or its chromium
binary) is missing, Tier 2 degrades to "unresolved" with a loud install hint — never a crash.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import analyze as AN
import download as DL

# How many hops past the seed links to follow (resource → doc → drive/link ≈ 3).
MAX_HOPS = 3

# A gate that stands between us and the footage → SKIP the campaign. Matched case-insensitively
# against fetched doc text / site page text. Liberal on purpose: intake only consults these to
# EXPLAIN a zero-footage outcome, so a header "log in" link on a page whose footage is freely
# reachable never triggers a skip (footage-found wins).
_BARRIER_PATTERNS = [
    ("login", [r"\blog[\s-]?in\b", r"\bsign[\s-]?in\b", r"\bsignin\b", r"login required",
               r"please log in", r"enter your password", r"members?[- ]only", r"authenticate to"]),
    ("signup", [r"\bsign[\s-]?up\b", r"\bsignup\b", r"create (an? )?account", r"create your account",
                r"register (now|to|for)", r"\bjoin (now|free|today)\b", r"become a member"]),
    ("payment", [r"\bpaywall\b", r"\bsubscribe\b", r"subscription required", r"\bpricing\b",
                 r"upgrade to (unlock|access|view|watch)", r"buy now", r"add to cart",
                 r"\bcheckout\b", r"\bpurchase\b", r"unlock (with|for) \$?\d", r"start (your )?(free )?trial"]),
    ("manual", [r"request access", r"email (us|me) (to|for)", r"\bdm (me|us)\b",
                r"apply (now|here|to)", r"fill out (the|this) form", r"contact (us|me) to"]),
]


def detect_barrier(text):
    """Return the barrier CATEGORY ('login'|'signup'|'payment'|'manual') if the text looks like a
    credential/payment/manual gate stands between us and the footage, else None."""
    low = (text or "").lower()
    for reason, pats in _BARRIER_PATTERNS:
        if any(re.search(p, low) for p in pats):
            return reason
    return None


def _is_footage_or_doc_link(u):
    """True if `u` is worth following: a footage source (Drive/YouTube/Kick/Twitch/direct video)
    or a Google Doc/Sheet. Used to filter the noise off a scraped page."""
    low = u.lower()
    if "drive.google.com" in low or "docs.google.com" in low:
        return True
    if any(h in low for h in ("youtube.com/watch", "youtu.be/", "youtube.com/shorts",
                              "youtube.com/@", "youtube.com/channel", "youtube.com/user",
                              "youtube.com/c/", "youtube.com/playlist", "list=",
                              "kick.com", "twitch.tv")):
        return True
    return os.path.splitext(u.split("?", 1)[0])[1].lower() in DL.VIDEO_EXTS


def _interesting_links(text):
    """Footage/doc links harvested from page HTML or anchor hrefs, deduped in order."""
    out, seen = [], set()
    for u in AN.harvest_urls(text or ""):
        if u not in seen and _is_footage_or_doc_link(u):
            seen.add(u)
            out.append(u)
    return out


def _short(e, n=160):
    return str(e).splitlines()[0][:n] if str(e) else ""


def playwright_available():
    """True if the Playwright library imports (the chromium binary is checked at launch time)."""
    try:
        from playwright.sync_api import sync_playwright  # noqa: F401
        return True
    except Exception:
        return False


_PW_INSTALL_HINT = ("pip install playwright && python -m playwright install chromium")


def _extract_with_playwright(url, timeout_ms=25000):
    """TIER 2: load a JS-rendered third-party page headlessly and scrape footage/doc links off it.
    Returns {links, barrier, tier:'browser', note}. Never raises — a missing chromium binary or a
    load error degrades to empty links + a note."""
    out = {"links": [], "barrier": None, "tier": "browser", "note": ""}
    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        out["note"] = f"Playwright not installed — {_PW_INSTALL_HINT}"
        return out
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                page = browser.new_page(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36")
                page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
                try:
                    page.wait_for_load_state("networkidle", timeout=8000)
                except Exception:
                    pass                                   # dynamic pages may never idle — proceed
                hrefs = page.eval_on_selector_all("a[href]", "els => els.map(e => e.href)") or []
                try:
                    body_text = page.inner_text("body")
                except Exception:
                    body_text = ""
                html = page.content()
            finally:
                browser.close()
        out["links"] = _interesting_links("\n".join(hrefs) + "\n" + html)
        out["barrier"] = detect_barrier(body_text) if not out["links"] else None
    except Exception as e:
        out["note"] = (f"browser load failed: {_short(e)} — is chromium installed? "
                       f"python -m playwright install chromium")
    return out


def extract_footage_links_from_site(url, cookies=None):
    """Resolve a THIRD-PARTY website to the footage/doc links on it, simplest-method-first.

    1) Try a plain HTTP fetch (Tier 1.5, no browser) — if the footage links are already in the
       served HTML, we're done without launching anything.
    2) Only if that yields NOTHING do we escalate to Playwright (Tier 2) to render the page.

    Returns {links: [...], barrier: 'login'|'signup'|'payment'|'manual'|None,
             tier: 'fetch'|'browser'|None, note: str}. `barrier` is only set when NO links were
    found (a page whose footage is freely reachable is never treated as gated)."""
    result = {"links": [], "barrier": None, "tier": None, "note": ""}

    html = None
    try:
        html = AN._fetch_text(url, timeout=25)
    except Exception as e:
        result["note"] = f"fetch failed: {_short(e)}"

    if html:
        links = _interesting_links(html)
        if links:
            result.update(links=links, tier="fetch")
            return result
        # No links in the static HTML — remember any gate hint before we try a browser.
        result["barrier"] = detect_barrier(html)

    pw = _extract_with_playwright(url)
    result["tier"] = "browser"
    result["links"] = pw["links"]
    result["note"] = pw["note"] or result["note"]
    # Prefer a barrier the rendered page reveals; fall back to the static-fetch hint.
    result["barrier"] = pw["barrier"] or (result["barrier"] if not pw["links"] else None)
    return result
