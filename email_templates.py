"""
email_templates.py — HTML renderers for SentiQuant transactional emails.

Holds the **watchlist update** email — dark theme, built to match
sentiquant.org's visual identity: pure-black ground (#000000), a subtly
lifted #0a0a0a content surface, #1a1a1a hairline dividers, white text,
Georgia serif for ticker symbols (approximating the site's Playfair Display
headline face), and the site's locked 3-state emerald / rose / gray badge
system. Brand blue (#0664e8) is used only for ₹ price values.

Table-based layout, inline styles + `bgcolor` attributes on every coloured
surface for older-Outlook safety.

------------------------------------------------------------------------------
INTEGRATION NOTE (watchlist email)
------------------------------------------------------------------------------
`render_watchlist_update_email()` expects a list of STRUCTURED update dicts:

    {"symbol": "HINDALCO.NSE", "type": "signal_change",
     "from": "HOLD/WATCH", "to": "STRONG BUY"}

    {"symbol": "HINDALCO.NSE", "type": "price_alert",
     "message": "price (₹1037.40) reached resistance R2 (₹1020.93)"}

main.py's `_detect_changes_for_item` still emits pre-formatted plain-text
`(kind, message)` tuples, so `_send_watchlist_email` renders via
`updates_from_legacy_msgs()` — a best-effort parser that converts the tuples
into the structured dicts. Migrating the detector to emit structured data
directly is still recommended (see handoff notes).
"""

from __future__ import annotations

import html
import re

# ── Palette ─────────────────────────────────────────────────────────────────
_BG_OUTER = "#000000"   # pure black wrapper
_BG_CARD = "#0a0a0a"    # lifted content surface
_DIVIDER = "#1a1a1a"    # 1px hairlines
_TEXT_PRIMARY = "#ffffff"
_TEXT_SECONDARY = "#9ca3af"
_TEXT_MUTED = "#6b7280"
_TEXT_DIM = "#4b5563"
_ARROW = "#4b5563"
_BLUE = "#0664e8"        # brand blue — ₹ values only
_AMBER = "#f59e0b"       # SEBI-disclaimer lead-in

# 3-state signal badge system (dark). Neutral also covers unrecognised.
_SIG_BULLISH = {"color": "#10b981", "bg": "#052e1f", "border": "#14532d"}
_SIG_BEARISH = {"color": "#f43f5e", "bg": "#2c0a12", "border": "#7f1d2e"}
_SIG_NEUTRAL = {"color": "#9ca3af", "bg": "#18181b", "border": "#27272a"}

# Serif for ticker headlines (Playfair Display stand-in); sans for everything else.
_SERIF = "Georgia,'Times New Roman',serif"
_SANS = "Arial,Helvetica,sans-serif"

_DISCLAIMER_LEAD = "AI-generated technical observations"
_DISCLAIMER = (
    f"These are {_DISCLAIMER_LEAD}, not investment advice. "
    "SentiQuant is not SEBI-registered. Past performance is not indicative of "
    "future results. Always do your own research."
)
_COPYRIGHT = "© 2026 SentiQuant · sentiquant.org"

# ₹ + digits (with optional thousands separators / decimals)
_PRICE_RE = re.compile(r"₹\s?[0-9][0-9,]*(?:\.[0-9]+)?")


# ── Helpers ──────────────────────────────────────────────────────────────────
def _split_symbol(raw: str):
    """'HINDALCO.NSE' -> ('HINDALCO', 'NSE'). Unknown/absent suffix -> (raw, '')."""
    s = (raw or "").strip().upper()
    for suffix in (".NSE", ".BSE"):
        if s.endswith(suffix):
            return s[: -len(suffix)], suffix[1:]
    if "." in s:
        base, _, ex = s.rpartition(".")
        return base, ex
    return s, ""


def _norm_signal(sig: str) -> str:
    return (sig or "").strip().upper()


def _signal_style(sig: str) -> dict:
    """Map a signal label to the 3-state badge palette."""
    s = _norm_signal(sig)
    if "BUY" in s:
        return _SIG_BULLISH
    if "SELL" in s:
        return _SIG_BEARISH
    return _SIG_NEUTRAL


def _signal_tag(text: str, style: dict) -> str:
    """The TO state — a sharp-cornered (2px) label, coloured tint + border per
    the 3-state system, letter-spaced so it reads as a printed label not a chip.
    No dot, no pill."""
    return (
        f'<span style="display:inline-block;padding:4px 10px;border-radius:2px;'
        f'background-color:{style["bg"]};color:{style["color"]};'
        f'border:1px solid {style["border"]};font-size:11px;font-weight:600;'
        f'font-family:{_SANS};white-space:nowrap;line-height:1.4;'
        f'letter-spacing:0.5px;text-transform:uppercase;">{html.escape(text)}</span>'
    )


def _exchange_badge(exchange: str) -> str:
    if not exchange:
        return ""
    return (
        f'<span style="display:inline-block;margin-top:6px;padding:2px 9px;'
        f'border-radius:9999px;background-color:{_DIVIDER};color:{_TEXT_SECONDARY};'
        f'font-size:10px;font-weight:600;font-family:{_SANS};letter-spacing:0.03em;">'
        f'{html.escape(exchange)}</span>'
    )


def _highlight_prices(text: str) -> str:
    """Escape text, then bold + brand-blue any ₹ price tokens."""
    escaped = html.escape(text)
    return _PRICE_RE.sub(
        lambda m: f'<span style="color:{_BLUE};font-weight:700;">{m.group(0)}</span>',
        escaped,
    )


def _symbol_headline(symbol_base: str) -> str:
    return (
        f'<span style="font-family:{_SERIF};font-size:17px;font-weight:700;'
        f'color:{_TEXT_PRIMARY};">{html.escape(symbol_base)}</span>'
    )


def _disclaimer_html() -> str:
    esc = html.escape(_DISCLAIMER)
    lead = html.escape(_DISCLAIMER_LEAD)
    return esc.replace(lead, f'<span style="color:{_AMBER};">{lead}</span>', 1)


# ── Row renderers ────────────────────────────────────────────────────────────
def _signal_change_row(symbol_base: str, exchange: str, sig_from: str, sig_to: str) -> str:
    # FROM — prior state: quiet plain text, no box, no dot, regular weight.
    from_txt = (f'<span style="color:{_TEXT_MUTED};font-size:11px;font-weight:400;'
                f'font-family:{_SANS};white-space:nowrap;">{html.escape(sig_from or "—")}</span>')
    # Arrow — a quiet connector, not a divider.
    arrow = (f'<span style="color:{_ARROW};font-size:11px;font-family:{_SANS};'
             f'padding:0 8px;">→</span>')
    # TO — new state: the one sharp-cornered coloured tag the eye lands on.
    to_tag = _signal_tag(sig_to or "—", _signal_style(sig_to))

    return f"""\
          <tr>
            <td style="padding:16px 32px;border-bottom:1px solid {_DIVIDER};background-color:{_BG_CARD};vertical-align:middle;" bgcolor="{_BG_CARD}">
              <table role="presentation" cellpadding="0" cellspacing="0" border="0" width="100%" style="border-collapse:collapse;">
                <tr>
                  <td align="left" style="vertical-align:middle;">
                    {_symbol_headline(symbol_base)}<br>
                    {_exchange_badge(exchange)}
                  </td>
                  <td align="right" style="vertical-align:middle;white-space:nowrap;font-family:{_SANS};">
                    {from_txt}{arrow}{to_tag}
                  </td>
                </tr>
              </table>
            </td>
          </tr>"""


def _price_alert_row(symbol_base: str, exchange: str, message: str) -> str:
    # Symbol first, exchange badge below it on its own line, then the message.
    header = ""
    if symbol_base:
        header = f"{_symbol_headline(symbol_base)}<br>"
        badge = _exchange_badge(exchange)
        if badge:
            header += f"{badge}<br>"

    msg_style = f"font-size:13px;color:{_TEXT_SECONDARY};line-height:1.6;"
    if header:
        msg_style += "display:inline-block;margin-top:8px;"

    return f"""\
          <tr>
            <td style="padding:16px 32px;border-bottom:1px solid {_DIVIDER};background-color:{_BG_CARD};font-family:{_SANS};" bgcolor="{_BG_CARD}">
              {header}<span style="{msg_style}">{_highlight_prices(message)}</span>
            </td>
          </tr>"""


def _render_rows(updates) -> str:
    rows = []
    for u in updates or []:
        symbol_base, exchange = _split_symbol(u.get("symbol", ""))
        utype = (u.get("type") or "").strip().lower()
        if utype in ("price_alert", "price"):
            rows.append(_price_alert_row(symbol_base, exchange, u.get("message", "")))
        else:  # signal_change (default)
            rows.append(_signal_change_row(
                symbol_base, exchange, u.get("from", ""), u.get("to", "")))
    if not rows:
        rows.append(
            f'          <tr><td style="padding:16px 32px;background-color:{_BG_CARD};'
            f'font-family:{_SANS};font-size:13px;color:{_TEXT_MUTED};" bgcolor="{_BG_CARD}">'
            'No changes to report.</td></tr>'
        )
    return "\n".join(rows)


# ── Public API ───────────────────────────────────────────────────────────────
def render_watchlist_update_email(updates) -> str:
    """Return the full HTML document for the watchlist update digest email.

    `updates` — list of dicts. Each dict:
        {"symbol": "<TICKER>.<NSE|BSE>", "type": "signal_change",
         "from": "<SIGNAL>", "to": "<SIGNAL>"}
      or
        {"symbol": "<TICKER>.<NSE|BSE>", "type": "price_alert",
         "message": "<full sentence, may contain ₹ values>"}
    """
    rows_html = _render_rows(updates)

    return f"""\
<!DOCTYPE html>
<html lang="en" xmlns="http://www.w3.org/1999/xhtml">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta http-equiv="X-UA-Compatible" content="IE=edge">
  <meta name="color-scheme" content="dark">
  <meta name="supported-color-schemes" content="dark">
  <title>SentiQuant watchlist update</title>
  <style>
    body {{ margin:0; padding:0; background:{_BG_OUTER}; -webkit-text-size-adjust:100%; -ms-text-size-adjust:100%; }}
    table {{ border-collapse:collapse; }}
    img {{ border:0; line-height:100%; outline:none; text-decoration:none; }}
    a {{ color:{_BLUE}; }}
    .sq-wrap {{ width:100%; background:{_BG_OUTER}; }}
    .sq-card {{ width:600px; max-width:600px; }}
    @media only screen and (max-width:620px) {{
      .sq-card {{ width:100% !important; }}
      .sq-row {{ padding-left:20px !important; padding-right:20px !important; }}
    }}
  </style>
</head>
<body style="margin:0;padding:0;background:{_BG_OUTER};font-family:{_SANS};">
  <!-- preheader -->
  <div style="display:none;max-height:0;overflow:hidden;mso-hide:all;font-size:1px;line-height:1px;color:{_BG_OUTER};">
    Technical changes on stocks you've analyzed.
  </div>

  <table role="presentation" class="sq-wrap" width="100%" cellpadding="0" cellspacing="0" border="0" style="background-color:{_BG_OUTER};" bgcolor="{_BG_OUTER}">
    <tr>
      <td align="center" style="padding:0;background-color:{_BG_OUTER};" bgcolor="{_BG_OUTER}">

        <table role="presentation" class="sq-card" width="600" cellpadding="0" cellspacing="0" border="0" style="width:600px;max-width:600px;margin:32px auto;">

          <!-- ── CARD (header + body) ──────────────────────────────────── -->
          <tr>
            <td style="background-color:{_BG_CARD};border:1px solid {_DIVIDER};border-radius:12px;overflow:hidden;" bgcolor="{_BG_CARD}">
              <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="width:100%;border-collapse:collapse;">

                <!-- header (black, no colored bar) -->
                <tr>
                  <td style="background-color:{_BG_OUTER};padding:24px 32px;border-bottom:1px solid {_DIVIDER};" bgcolor="{_BG_OUTER}">
                    <div style="font-family:{_SERIF};color:{_TEXT_PRIMARY};font-size:20px;font-weight:700;line-height:1.4;">
                      SentiQuant
                    </div>
                    <div style="font-family:{_SANS};color:{_TEXT_MUTED};font-size:13px;line-height:1.4;margin-top:3px;">
                      Watchlist update
                    </div>
                  </td>
                </tr>

                <!-- intro -->
                <tr>
                  <td class="sq-row" style="padding:20px 32px 0 32px;background-color:{_BG_CARD};" bgcolor="{_BG_CARD}">
                    <p style="margin:0 0 16px 0;color:{_TEXT_SECONDARY};font-size:13px;line-height:1.5;font-family:{_SANS};">
                      Technical changes on stocks you've analyzed:
                    </p>
                  </td>
                </tr>

                <!-- rows -->
                <tr>
                  <td style="background-color:{_BG_CARD};padding:0 0 4px 0;" bgcolor="{_BG_CARD}">
                    <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="width:100%;border-collapse:collapse;">
{rows_html}
                    </table>
                  </td>
                </tr>

              </table>
            </td>
          </tr>

          <!-- ── FOOTER (outside the card, pure black) ─────────────────── -->
          <tr>
            <td style="background-color:{_BG_OUTER};padding:24px 32px;border-top:1px solid {_DIVIDER};" bgcolor="{_BG_OUTER}">
              <p style="margin:0 0 8px 0;color:{_TEXT_MUTED};font-size:11px;line-height:1.6;text-align:center;font-family:{_SANS};">
                {_disclaimer_html()}
              </p>
              <p style="margin:0;color:{_TEXT_DIM};font-size:10px;line-height:1.5;text-align:center;font-family:{_SANS};">
                {html.escape(_COPYRIGHT)}
              </p>
            </td>
          </tr>

        </table>

      </td>
    </tr>
  </table>
</body>
</html>"""


# ── Interim shim: current (kind, message) tuples -> structured updates ────────
_LEGACY_SIGNAL_RE = re.compile(
    r"^(?P<sym>\S+):\s.*?\((?P<from>.+?)\s*(?:→|->)\s*(?P<to>.+?)\)\.?\s*$"
)


def updates_from_legacy_msgs(msgs):
    """Best-effort convert the current main.py `(kind, message)` tuples into the
    structured dicts this template expects. Use ONLY until main.py is updated to
    emit structured data directly.

    - 'signal_flip' -> parse 'SYM: ... (FROM -> TO).' into a signal_change dict
    - everything else -> price_alert with the raw sentence (SYM: stripped)
    """
    out = []
    for entry in (msgs or []):
        # ── Defensive unpacking — tolerate bare strings / malformed entries ──
        if isinstance(entry, (list, tuple)) and len(entry) == 2:
            kind, message = entry
        elif isinstance(entry, str):
            kind, message = None, entry
        else:
            out.append({"symbol": "", "type": "price_alert", "message": str(entry)})
            continue

        message = (message or "").strip()
        if not message:
            continue

        # ── Signal flips: "SYM: ... (FROM → TO)."  (kinds: signal_flip) ──
        if kind == "signal_flip":
            m = _LEGACY_SIGNAL_RE.match(message)
            if m:
                out.append({
                    "symbol": m.group("sym"),
                    "type": "signal_change",
                    "from": m.group("from").strip(),
                    "to": m.group("to").strip(),
                })
                continue
            # unparseable flip -> fall through to a plain-text row

        # ── Price alerts (stop_hit / target_hit), unknown kinds, bad flips ──
        # Real formats:
        #   "SYM: price (₹X) reached your stop-loss reference level (₹Y)."
        #   "SYM: price (₹X) reached resistance reference R2 (₹Y)."
        sym = ""
        body = message
        if ":" in message:
            head, _, rest = message.partition(":")
            # a bare ticker: no spaces, no currency symbol
            if head and " " not in head and "₹" not in head:
                sym, body = head.strip(), rest.strip()
        out.append({"symbol": sym, "type": "price_alert", "message": body})
    return out
