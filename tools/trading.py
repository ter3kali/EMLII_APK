"""
!trade — TERMSIM, the stock-trading simulator, as an EMLII tool.

Each EMLII user gets their own portfolio, saved in data/trading/<username>.json.

    !trade                      help + your net worth
    !trade prices               all tickers with change and trend
    !trade chart NEBX           ASCII price chart
    !trade buy NEBX 10          buy shares   (qty can be "max")
    !trade sell NEBX 5          sell shares  (qty can be "all")
    !trade port                 holdings, P/L and net worth
    !trade news                 today's headlines
    !trade next [days]          advance the market 1+ trading days
    !trade watch [NEBX] [ticks] the old "live" view: run N intraday ticks and
                                chart the session (omit ticker = net worth)
    !trade import               load an old ~/.termsim_save.json
    !trade reset confirm        start over with $10,000

Standard library only.
"""
import json
import os
import random
import re
import threading
from pathlib import Path

COMMAND = "trade"
HELP = "Stock trading sim: !trade prices|buy|sell|port|next|news|chart"

STARTING_CASH = 10000.0


def _save_dir():
    return Path(os.environ.get("EMLII_DATA_DIR") or "data") / "trading"


LEGACY_SAVE = os.path.expanduser("~/.termsim_save.json")
_lock = threading.RLock()

STARTING_MARKET = {
    "NEBX": {"price": 142.50, "sector": "Tech", "vol": 0.020},
    "QBIT": {"price": 88.10, "sector": "Tech", "vol": 0.030},
    "SOLR": {"price": 34.75, "sector": "Energy", "vol": 0.025},
    "PTRL": {"price": 61.20, "sector": "Energy", "vol": 0.018},
    "VBNK": {"price": 210.00, "sector": "Finance", "vol": 0.012},
    "CRDX": {"price": 45.30, "sector": "Finance", "vol": 0.015},
    "MEDV": {"price": 97.60, "sector": "Health", "vol": 0.014},
    "GENZ": {"price": 22.40, "sector": "Health", "vol": 0.022},
    "SHPX": {"price": 15.80, "sector": "Retail", "vol": 0.020},
    "URBN": {"price": 53.15, "sector": "Retail", "vol": 0.017},
    "DOGX": {"price": 0.85, "sector": "Crypto", "vol": 0.080},
    "LUNR": {"price": 12.40, "sector": "Crypto", "vol": 0.070},
}

NEWS_EVENTS = [
    ("Tech", +1, "A surprise chip breakthrough boosts investor confidence in Tech."),
    ("Tech", -1, "Regulators announce a new antitrust probe into Tech giants."),
    ("Energy", +1, "Oil prices spike after supply disruption fears."),
    ("Energy", -1, "New renewable subsidies pressure traditional Energy names."),
    ("Finance", +1, "Central bank hints at rate cuts, lifting Finance stocks."),
    ("Finance", -1, "Credit downgrade warning rattles the Finance sector."),
    ("Health", +1, "A major drug trial reports unexpectedly strong results."),
    ("Health", -1, "Patent litigation clouds outlook for Health companies."),
    ("Retail", +1, "Holiday sales figures crush analyst expectations."),
    ("Retail", -1, "Consumer spending data comes in weaker than forecast."),
    ("Crypto", +1, "A major exchange announces new institutional adoption."),
    ("Crypto", -1, "A large hack triggers a sell-off across Crypto tokens."),
    (None, +1, "Broad market rallies on cooling inflation data."),
    (None, -1, "Markets slide amid recession fears."),
]


# ── state ────────────────────────────────────────────────────────────────────
def new_game():
    market = {t: dict(v, history=[v["price"]]) for t, v in STARTING_MARKET.items()}
    return {"cash": STARTING_CASH, "holdings": {}, "avg_cost": {}, "market": market,
            "day": 1, "net_worth_history": [STARTING_CASH], "last_news": [], "realized_pnl": 0.0}


def _save_path(user):
    safe = re.sub(r"[^\w.-]", "_", user or "") or "player"
    return _save_dir() / f"{safe}.json"


def load_game(user):
    p = _save_path(user)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return new_game()


def save_game(user, state):
    p = _save_path(user)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(state))
    tmp.replace(p)                                   # atomic: never leaves a half-written save


# ── market simulation (ported unchanged) ─────────────────────────────────────
def _apply_move(info, pct):
    info["price"] = round(max(0.05, info["price"] * (1 + pct)), 2)
    info["history"].append(info["price"])
    if len(info["history"]) > 200:
        info["history"] = info["history"][-200:]


def net_worth(state):
    total = state["cash"]
    for t, qty in state["holdings"].items():
        total += qty * state["market"][t]["price"]
    return round(total, 2)


def advance_day(state):
    state["day"] += 1
    events = []
    if random.random() < 0.55:
        sector, direction, headline = random.choice(NEWS_EVENTS)
        events.append(headline)
        for info in state["market"].values():
            if sector is None or info["sector"] == sector:
                _apply_move(info, direction * random.uniform(0.03, 0.09))
    for info in state["market"].values():
        _apply_move(info, random.uniform(-info["vol"], info["vol"]))
    state["last_news"] = events
    state["net_worth_history"].append(net_worth(state))
    state["net_worth_history"] = state["net_worth_history"][-200:]
    return events


# ── trading (return text instead of printing) ────────────────────────────────
def buy(state, ticker, qty):
    ticker = ticker.upper()
    if ticker not in state["market"]:
        return f"No such ticker: {ticker}"
    price = state["market"][ticker]["price"]
    if qty == "max":
        qty = int(state["cash"] // price)
        if qty < 1:
            return f"Not enough cash for even one share (${price:,.2f})."
    cost = price * qty
    if cost > state["cash"]:
        return f"Not enough cash. Need ${cost:,.2f}, have ${state['cash']:,.2f}"
    prev_qty = state["holdings"].get(ticker, 0)
    prev_avg = state["avg_cost"].get(ticker, 0.0)
    new_qty = prev_qty + qty
    state["avg_cost"][ticker] = round((prev_qty * prev_avg + qty * price) / new_qty, 2)
    state["holdings"][ticker] = new_qty
    state["cash"] = round(state["cash"] - cost, 2)
    return f"Bought {qty} {ticker} @ ${price:,.2f} = ${cost:,.2f}\nCash left: ${state['cash']:,.2f}"


def sell(state, ticker, qty):
    ticker = ticker.upper()
    have = state["holdings"].get(ticker, 0)
    if qty == "all":
        qty = have
    if not have or have < qty or qty < 1:
        return f"You only hold {have} shares of {ticker}"
    price = state["market"][ticker]["price"]
    proceeds = price * qty
    avg = state["avg_cost"].get(ticker, price)
    pnl = (price - avg) * qty
    state["realized_pnl"] = round(state["realized_pnl"] + pnl, 2)
    state["holdings"][ticker] -= qty
    if state["holdings"][ticker] == 0:
        del state["holdings"][ticker]
        state["avg_cost"].pop(ticker, None)
    state["cash"] = round(state["cash"] + proceeds, 2)
    return (f"Sold {qty} {ticker} @ ${price:,.2f} = ${proceeds:,.2f}  (P/L {_money(pnl, True)})\n"
            f"Cash: ${state['cash']:,.2f}")


# ── formatting (plain text, so it looks right in the terminal, desktop and web UIs) ─
def _money(v, sign=False):
    return f"{'+' if v >= 0 else '-'}${abs(v):,.2f}" if sign else f"${v:,.2f}"


def _pct(v):
    return f"{'▲' if v > 0 else '▼' if v < 0 else '·'}{abs(v):.2f}%"


def sparkline(values, width=12):
    if len(values) < 2:
        return ""
    values = values[-width:]
    blocks = "▁▂▃▄▅▆▇█"
    lo, hi = min(values), max(values)
    span = hi - lo or 1
    return "".join(blocks[int((v - lo) / span * (len(blocks) - 1))] for v in values)


def show_prices(state):
    out = [f"Day {state['day']} · Market", f"{'TICK':<6}{'SECTOR':<8}{'PRICE':>9}{'CHG':>9}  TREND"]
    for t, info in state["market"].items():
        h = info["history"]
        chg = (h[-1] - h[-2]) / h[-2] * 100 if len(h) >= 2 and h[-2] else 0.0
        out.append(f"{t:<6}{info['sector']:<8}{info['price']:>9,.2f}{_pct(chg):>9}  {sparkline(h)}")
    return "\n".join(out)


def show_portfolio(state):
    nw = net_worth(state)
    out = [f"Portfolio · Day {state['day']}", f"Cash: ${state['cash']:,.2f}"]
    if state["holdings"]:
        out.append(f"\n{'TICK':<6}{'QTY':>5}{'AVG':>9}{'PRICE':>9}{'VALUE':>11}{'P/L':>11}")
        for t, qty in state["holdings"].items():
            price, avg = state["market"][t]["price"], state["avg_cost"][t]
            out.append(f"{t:<6}{qty:>5}{avg:>9,.2f}{price:>9,.2f}{price * qty:>11,.2f}"
                       f"{(price - avg) * qty:>+11,.2f}")
    else:
        out.append("\n(no holdings)")
    out.append(f"\nRealized P/L: {_money(state['realized_pnl'], True)}")
    out.append(f"Net worth: ${nw:,.2f}  (started ${STARTING_CASH:,.0f}, "
               f"{_pct((nw - STARTING_CASH) / STARTING_CASH * 100)})")
    out.append(f"History: {sparkline(state['net_worth_history'], 30)}")
    return "\n".join(out)


def _chart(label, data, start_val, rows=10, note=""):
    data = data[-40:]
    lo, hi = min(data), max(data)
    span = (hi - lo) or max(abs(hi), 1) * 0.001
    cur = data[-1]
    chg = (cur - start_val) / start_val * 100 if start_val else 0.0
    out = [f"{label}   ${cur:,.2f}   {_pct(chg)}   hi ${hi:,.2f} / lo ${lo:,.2f}", ""]
    for r in range(rows, -1, -1):
        thr = lo + span * (r / rows)
        row = "".join(("●" if i == len(data) - 1 else "│") if v >= thr else " "
                      for i, v in enumerate(data))
        out.append(f"{thr:>8.2f} {row}")
    out.append(" " * 9 + "-" * len(data))
    if note:
        out.append(note)
    return "\n".join(out)


def show_chart(state, ticker):
    t = ticker.upper()
    if t not in state["market"]:
        return f"No such ticker: {t}"
    h = state["market"][t]["history"]
    if len(h) < 2:
        return f"{t} has no history yet — try !trade next first."
    return _chart(f"{t} price history", h, h[0])


def show_news(state):
    if not state["last_news"]:
        return "No major news today."
    return "Headlines\n" + "\n".join(f"  • {h}" for h in state["last_news"])


def watch(state, ticker=None, ticks=40):
    """The old live view, as a batch: advance `ticks` intraday moves, then chart the session."""
    if ticker:
        ticker = ticker.upper()
        if ticker not in state["market"]:
            return f"No such ticker: {ticker}"
        start, label = state["market"][ticker]["price"], f"{ticker} — session"
    else:
        start, label = net_worth(state), "Net worth — session"
    session = [start]
    for _ in range(ticks):
        if ticker:
            info = state["market"][ticker]
            _apply_move(info, random.uniform(-info["vol"] / 3, info["vol"] / 3))
            session.append(info["price"])
        else:
            for info in state["market"].values():
                _apply_move(info, random.uniform(-info["vol"] / 3, info["vol"] / 3))
            session.append(net_worth(state))
    return _chart(label, session, start, rows=12, note=f"  {ticks} ticks simulated")


HELP_TEXT = """TERMSIM — trading simulator
  !trade prices            show all prices
  !trade chart <TICKER>    price chart
  !trade buy <TICKER> <QTY|max>
  !trade sell <TICKER> <QTY|all>
  !trade port              holdings & net worth
  !trade news              today's headlines
  !trade next [days]       advance the market
  !trade watch [TICKER] [ticks]   simulate a live session
  !trade import / reset confirm"""


# ── EMLII entry point ────────────────────────────────────────────────────────
def _reply(text):
    return {"type": "reply", "text": text}


def _qty(word, special):
    if word.lower() == special:
        return special
    q = int(word)                       # ValueError → usage message
    if q <= 0:
        raise ValueError
    return q


def run(arg, ctx):
    user = (ctx or {}).get("username") or "player"
    parts = (arg or "").split()
    cmd = parts[0].lower() if parts else ""
    rest = parts[1:]

    with _lock:                                       # GUI + web can call us at the same time
        state = load_game(user)
        changed = False
        try:
            if cmd in ("", "help", "h", "?"):
                text = HELP_TEXT + f"\n\nDay {state['day']} · Net worth ${net_worth(state):,.2f}"
                if state["day"] == 1 and not state["holdings"]:
                    text += f"\nStarting cash: ${state['cash']:,.2f}"
            elif cmd in ("prices", "price", "p", "market"):
                text = show_prices(state)
            elif cmd in ("chart", "c"):
                text = show_chart(state, rest[0]) if rest else "Usage: !trade chart <TICKER>"
            elif cmd in ("port", "portfolio", "pf"):
                text = show_portfolio(state)
            elif cmd == "news":
                text = show_news(state)
            elif cmd in ("buy", "sell"):
                if len(rest) < 2:
                    return _reply(f"Usage: !trade {cmd} <TICKER> <{'QTY|max' if cmd == 'buy' else 'QTY|all'}>")
                try:
                    qty = _qty(rest[1], "max" if cmd == "buy" else "all")
                except ValueError:
                    return _reply(f"Usage: !trade {cmd} <TICKER> <QTY>  (a whole number above 0)")
                text = (buy if cmd == "buy" else sell)(state, rest[0], qty)
                changed = text.startswith(("Bought", "Sold"))
            elif cmd in ("next", "n"):
                days = max(1, min(int(rest[0]), 30)) if rest and rest[0].isdigit() else 1
                headlines = []
                for _ in range(days):
                    headlines += advance_day(state)
                changed = True
                state["last_news"] = headlines[-3:] if days > 1 else headlines
                text = (f"=== Day {state['day']} begins ===" +
                        (f"  ({days} days passed)" if days > 1 else "") +
                        "\n" + show_news(state) + "\n\n" + show_prices(state))
            elif cmd in ("watch", "live"):
                ticker = rest[0] if rest and not rest[0].isdigit() else None
                nums = [w for w in rest if w.isdigit()]
                ticks = max(5, min(int(nums[0]), 300)) if nums else 40
                text = watch(state, ticker, ticks)
                changed = not text.startswith("No such")
            elif cmd == "import":
                if not os.path.exists(LEGACY_SAVE):
                    return _reply(f"No old save found at {LEGACY_SAVE}")
                with open(LEGACY_SAVE) as f:
                    state = json.load(f)
                changed = True
                text = f"Imported your old TERMSIM save (day {state.get('day', '?')})."
            elif cmd == "reset":
                if rest and rest[0].lower() == "confirm":
                    state, changed = new_game(), True
                    text = f"Fresh start — ${STARTING_CASH:,.0f} cash, day 1."
                else:
                    text = "This erases your portfolio. Type  !trade reset confirm  to continue."
            else:
                text = f"Unknown option '{cmd}'.\n\n" + HELP_TEXT
        except (KeyError, IndexError, ValueError) as e:
            return _reply(f"Something went wrong ({e}). Try !trade reset confirm if the save is damaged.")
        if changed:
            save_game(user, state)
        return _reply(text)
