#!/usr/bin/env python3


import os, re, io, sys, csv, json, time, base64, html, shutil, hashlib
import smtplib, argparse, unicodedata
from email.message import EmailMessage
from datetime import datetime, timedelta, timezone
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests, cv2
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from PIL import Image, ImageOps, ImageStat

# ----------------------------- CONFIG ---------------------------------------
ZENPUT_TOKEN = os.environ.get("ZENPUT_TOKEN", "")
WEIGHTS = os.environ.get("WEIGHTS", "models/valve_v2_best.pt")

SMTP_HOST, SMTP_PORT = "smtp.gmail.com", 465
SMTP_USER = os.environ.get("SMTP_USER", "aof.group.auto@gmail.com")
SMTP_PASS = os.environ.get("SMTP_PASS", "")
SENDER_NAME = "Business Intelligence"

# --- recipients: edit these lists, add as many addresses as you like ----------
MAIL_TO = [
       "w.alhanani@aofgroup.com","m.alsaghir@aofgroup.com","n.joshe@aofgroup.com","a.banafe@aofgroup.com",
    "i.mostafa@aofgroup.com","m.alghazali@aofgroup.com","s.poudel@aofgroup.com","m.emad@aofgroup.com",
    "a.suliman@aofgroup.com","a.alarabi@aofgroup.com","s.mansuri@aofgroup.com","m.suhail@aofgroup.com","a.alghanimi@lubdasa.com"
]

MAIL_CC = [
   "o.salahaddin@aofgroup.com","a.alsalem@aofgroup.com","m.hejazi@aofgroup.com",
    "omar@aofgroup.com","m.alhuaydar@aofgroup.com",
    "a.omara@aofgroup.com","s.alharbi@aofgroup.com",
]
# who gets the review copy each morning (nobody else is emailed until you send)
MAIL_REVIEWER = [
    "o.salahaddin@aofgroup.com",
]

TEMPLATES = {"Classic": 401648, "Lubda": 472189, "Garatis": 671643}

# 0 = today (KSA), 1 = yesterday. Closing checklists are submitted late at night,
# so a run before midnight sees today's; a run the next morning sees yesterday's.
DAYS_BACK = 0

# 24-hour branches that do not close the gas — reported separately, not scored.
EXCLUDE_BRANCHES = {"B22", "B28", "B30", "B33", "QB04", "QB05", "QB07"}

IMGSZ, CONF_FLOOR, CONF_OPEN = 1280, 0.30, 0.20
EXPECTED_VALVES, IOU, AGNOSTIC = 3, 0.45, True
TWO_PASS, CLAHE_CLIP, CONF_PASS2, MERGE_IOU = True, 3.0, 0.35, 0.40
DARK_MEAN, DARK_STD = 25, 15
MAX_SIDE, JPEG_Q, THUMB_MAX, THUMB_Q = 1280, 88, 900, 78
PAGE_LIMIT, MAX_PAGES, STALE_PAGES, WORKERS = 50, 200, 3, 6

BASE = "https://www.zenput.com"
WORK = "/tmp/gasbox"
VERDICTS = ["correct", "WRONG", "REVIEW", "UNUSABLE"]
REASON = {"correct":  "all three closed",
          "WRONG":    "open valve",
          "REVIEW":   "needs a human check",
          "UNUSABLE": "photo does not show the gas box"}
# -----------------------------------------------------------------------------

sess = requests.Session()
sess.headers.update({"X-API-TOKEN": ZENPUT_TOKEN})
sess.mount("https://", HTTPAdapter(
    max_retries=Retry(total=5, backoff_factor=1.5,
                      status_forcelist=[429, 500, 502, 503, 504],
                      allowed_methods=["GET"]),
    pool_connections=8, pool_maxsize=8))


def get_json(url, params, tries=4):
    last = None
    for a in range(tries):
        try:
            r = sess.get(url, params=params, timeout=(15, 120))
            r.raise_for_status(); return r.json()
        except Exception as e:
            last = e; time.sleep(2 ** a)
    raise last


def norm(s):
    if not s: return ""
    s = unicodedata.normalize("NFKC", str(s)).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^0-9a-z\u0600-\u06FF]+", " ", s)).strip()


def is_gasbox(t):
    n = norm(t)
    return (("gas box" in n and "inside" in n) or
            (("بوكس الغاز" in n or "صندوق الغاز" in n) and "افتح" in n))


CODE_RE = re.compile(r"(?<![A-Za-z])(QB|LB|B)\s*-?\s*(\d{1,3})(?![0-9])")


def branch_code(name, lid):
    if name:
        best = None
        for m in CODE_RE.finditer(str(name)):
            pre, num = m.group(1).upper(), int(m.group(2))
            cand = f"{pre}{num:02d}"
            if pre in ("QB", "LB"): return cand
            best = best or cand
        if best: return best
    return f"LOC{lid}" if lid else "UNK"


def parse_day(v):
    m = re.search(r"\d{4}-\d{2}-\d{2}", str(v or "")); return m.group(0) if m else ""


def deep_url(o):
    if isinstance(o, str): return o if o.startswith("http") else None
    if isinstance(o, dict):
        for k in ("url", "signed_url", "signedUrl", "link", "href"):
            if k in o and (u := deep_url(o[k])): return u
        for v in o.values():
            if u := deep_url(v): return u
    if isinstance(o, list):
        for v in o:
            if u := deep_url(v): return u
    return None


def signed(k):
    return deep_url(get_json(f"{BASE}/api/v2/users/current/storage/", {"path": k}))


def fetch(tid, day):
    out, start, stale = [], 0, 0
    for _ in range(MAX_PAGES):
        j = get_json(f"{BASE}/api/v3/submissions/",
                     {"form_template_id": tid, "limit": PAGE_LIMIT, "start": start})
        batch = j if isinstance(j, list) else (j.get("data") or j.get("results") or [])
        if not batch: break
        out.extend(batch)
        days = [d for d in (parse_day((s.get("smetadata") or {}).get("date_submitted"))
                            for s in batch) if d]
        stale = stale + 1 if (days and max(days) < day) else 0
        if stale >= STALE_PAGES or len(batch) < PAGE_LIMIT: break
        start += PAGE_LIMIT
    return out


def collect(day):
    recs = []
    for brand, tid in TEMPLATES.items():
        try:
            subs = fetch(tid, day)
        except Exception as e:
            print(f"  !! {brand} fetch failed: {e}"); continue
        n = 0
        for s in subs:
            meta = s.get("smetadata") or {}
            if parse_day(meta.get("date_submitted")) != day: continue
            loc = meta.get("location") or {}
            lname = loc.get("name") if isinstance(loc, dict) else str(loc)
            lid = loc.get("id") if isinstance(loc, dict) else None
            for a in s.get("answers") or []:
                if a.get("field_type") != "image" or not is_gasbox(a.get("title")): continue
                ph = a.get("value") or []
                if isinstance(ph, dict): ph = [ph]
                for p_ in ph:
                    if isinstance(p_, dict) and p_.get("s3_key"):
                        recs.append({"brand": brand, "branch": branch_code(lname, lid),
                            "branch_name": (lname or "").replace("\t", " ").strip(),
                            "date": day, "sid": s.get("id", ""),
                            "user": (meta.get("created_by") or {}).get("display_name", ""),
                            "s3_key": p_["s3_key"]})
                        n += 1
                break
        print(f"  {brand}: {n} gas box photos")
    seen, uniq, excluded = set(), [], []
    for r in recs:
        if r["s3_key"] in seen: continue
        seen.add(r["s3_key"])
        (excluded if r["branch"] in EXCLUDE_BRANCHES else uniq).append(r)
    if excluded:
        eb = sorted({r["branch"] for r in excluded})
        print(f"  excluded {len(excluded)} photos from 24h branches: {', '.join(eb)}")
    # sort by s3_key so indices are stable between stage 1 and stage 2
    return sorted(uniq, key=lambda r: r["s3_key"]), sorted(
        excluded, key=lambda r: (r["brand"], r["branch"]))


def fingerprint(records):
    h = hashlib.sha1("".join(r["s3_key"] for r in records).encode()).hexdigest()
    return h[:4]


def enhance(src, dst):
    im = cv2.imread(src)
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(8, 8)).apply(l)
    cv2.imwrite(dst, cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR))


def download(records):
    def grab(ir):
        i, r = ir
        try:
            u = signed(r["s3_key"])
            if not u: return r, None, "no url", None
            raw = sess.get(u, timeout=120).content
            if len(raw) < 5000: return r, None, "tiny", None
            im = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
            if max(im.size) > MAX_SIDE: im.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
            fn = f"{i:03d}__{r['brand']}__{r['branch']}.jpg"
            im.save(f"{WORK}/img/{fn}", "JPEG", quality=JPEG_Q, optimize=True)
            st = ImageStat.Stat(im.convert("L"))
            if TWO_PASS: enhance(f"{WORK}/img/{fn}", f"{WORK}/enh/{fn}")
            return r, fn, None, {"mean": round(st.mean[0], 1),
                                 "std": round(st.stddev[0], 1), "idx": i}
        except Exception as e:
            return r, None, str(e)[:90], None

    got, fails = [], []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for f in as_completed([ex.submit(grab, (i, r)) for i, r in enumerate(records)]):
            r, fn, err, stats = f.result()
            if err: fails.append((r, err))
            else: got.append({**r, "file": fn, **stats})
    return sorted(got, key=lambda g: g["idx"]), fails


def iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0: return 0.0
    return inter / ((ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter)


def score(got):
    from ultralytics import YOLO
    model = YOLO(WEIGHTS)
    names = [model.names[i] for i in sorted(model.names)]
    OPEN = next(i for i, n in enumerate(names) if "open" in n.lower())

    def detect(path, floor):
        r = model.predict(path, imgsz=IMGSZ, conf=floor, iou=IOU,
                          agnostic_nms=AGNOSTIC, verbose=False)[0]
        return [(int(c), float(p), xy) for c, p, xy in
                zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist())]

    rows = []
    for g in got:
        base = [(c, p, xy, "=") for c, p, xy in
                detect(f"{WORK}/img/{g['file']}", min(CONF_FLOOR, CONF_OPEN))
                if p >= (CONF_OPEN if c == OPEN else CONF_FLOOR)]
        extra = []
        if TWO_PASS and os.path.exists(f"{WORK}/enh/{g['file']}"):
            for c, p, xy in detect(f"{WORK}/enh/{g['file']}", min(CONF_PASS2, CONF_OPEN)):
                if p < (CONF_OPEN if c == OPEN else CONF_PASS2): continue
                if any(iou(xy, b[2]) > MERGE_IOU for b in base + extra): continue
                extra.append((c, p, xy, "+"))

        cands = sorted(base + extra, key=lambda t: -t[1])
        keep, dropped = cands[:EXPECTED_VALVES], cands[EXPECTED_VALVES:]
        cls = [c for c, _, _, _ in keep]
        dark = g["mean"] < DARK_MEAN and g["std"] < DARK_STD
        if len(cls) == 0:
            v, why = "UNUSABLE", ("black/blank frame" if dark else "no valves visible")
        elif len(cls) < EXPECTED_VALVES:
            v, why = "REVIEW", f"only {len(cls)} of {EXPECTED_VALVES} valves found"
        elif OPEN in cls:
            v, why = "WRONG", f"{cls.count(OPEN)} open valve(s)"
        else:
            v, why = "correct", "all three closed"

        im = cv2.imread(f"{WORK}/img/{g['file']}")
        for c, p, xy, src in keep:
            x1, y1, x2, y2 = [int(t) for t in xy]
            col = (0,0,220) if c == OPEN else ((0,170,255) if src == "+" else (255,90,0))
            cv2.rectangle(im, (x1,y1), (x2,y2), col, 3)
            cv2.putText(im, f"{src}{names[c][6:]} {p:.2f}", (x1, max(18, y1-7)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
        for c, p, xy, _ in dropped:
            x1, y1, x2, y2 = [int(t) for t in xy]
            cv2.rectangle(im, (x1,y1), (x2,y2), (150,150,150), 1)
        out = f"{WORK}/out/{g['idx']:03d}.jpg"
        cv2.imwrite(out, im)

        rows.append({"idx": g["idx"], "brand": g["brand"], "branch": g["branch"],
            "branch_name": g["branch_name"], "user": g["user"], "date": g["date"],
            "sid": g["sid"], "auto_verdict": v, "verdict": v, "reason": why,
            "corrected": False, "n_valves": len(cls), "n_open": cls.count(OPEN),
            "n_dropped": len(dropped), "mean_lum": g["mean"], "img": out,
            "detections": " ".join(f"{s}{names[c][6:]}:{p:.2f}" for c,p,_,s in keep),
            "dropped": " ".join(f"{names[c][6:]}:{p:.2f}" for c,p,_,_ in dropped)})
    return rows


def b64(path):
    im = Image.open(path).convert("RGB")
    if max(im.size) > THUMB_MAX: im.thumbnail((THUMB_MAX, THUMB_MAX), Image.LANCZOS)
    b = io.BytesIO(); im.save(b, "JPEG", quality=THUMB_Q, optimize=True)
    return base64.b64encode(b.getvalue()).decode()


SECTIONS = [("WRONG", "Open valve detected", "#E24B4A"),
            ("REVIEW", "Model unsure — human check", "#D9822B"),
            ("UNUSABLE", "Photo does not show the gas box", "#6B6B6B"),
            ("correct", "All three closed", "#1D9E75")]
COLOR = {k: c for k, _, c in SECTIONS}


def build_html(rows, day, fp, review_mode, n_fail=0):
    c = Counter(r["verdict"] for r in rows)
    usable = len(rows) - c.get("UNUSABLE", 0)
    ncorr = sum(1 for r in rows if r["corrected"])

    css = """body{font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:0;
background:#f4f4f2;color:#1a1a18;padding-bottom:70px}
header{background:#1a1a18;color:#fff;padding:22px 28px}header h1{margin:0;font-size:20px}
header p{margin:4px 0 0;opacity:.7;font-size:13px}
.totals{display:flex;gap:10px;flex-wrap:wrap;padding:18px 28px}
.pill{padding:10px 16px;border-radius:8px;color:#fff;font-size:14px;font-weight:600}
.filters{padding:0 28px 8px}
.filters button{border:1px solid #ccc;background:#fff;border-radius:6px;
padding:7px 14px;margin-right:8px;cursor:pointer;font-size:13px}
.filters button.on{background:#1a1a18;color:#fff}
h2{margin:26px 28px 10px;font-size:16px;display:flex;align-items:center;gap:10px}
.dot{width:10px;height:10px;border-radius:50%;display:inline-block}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));
gap:16px;padding:0 28px 10px}
.card{background:#fff;border-radius:10px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,.12)}
.card.ed{outline:3px solid #2F6FEB}
.card img{width:100%;display:block;background:#000;cursor:zoom-in}
.meta{padding:11px 13px;font-size:13px;line-height:1.5}
.meta .b{font-weight:600}.meta .sub{color:#666;font-size:12px}
.tag{display:inline-block;padding:2px 8px;border-radius:4px;color:#fff;
font-size:11px;font-weight:600}
.why{font-size:12px;color:#333;margin-top:4px}
.det{font-family:ui-monospace,Menlo,monospace;font-size:11px;color:#444;margin-top:5px}
.drp{font-family:ui-monospace,Menlo,monospace;font-size:11px;color:#999}
.legend{padding:0 28px 14px;color:#666;font-size:12px;line-height:1.7}
.fix{display:flex;gap:5px;flex-wrap:wrap;margin-top:9px;border-top:1px solid #eee;
padding-top:9px}
.fix button{border:1px solid #ccc;background:#fafafa;border-radius:5px;padding:4px 9px;
font-size:11px;cursor:pointer}
.fix button.sel{color:#fff;border-color:transparent;font-weight:600}
.bar{position:fixed;left:0;right:0;bottom:0;background:#1a1a18;color:#fff;
padding:13px 28px;display:flex;align-items:center;gap:16px;font-size:14px;
box-shadow:0 -2px 10px rgba(0,0,0,.25)}
.bar button{background:#2F6FEB;color:#fff;border:0;border-radius:6px;padding:9px 18px;
font-size:14px;cursor:pointer;font-weight:600}
.bar .out{font-family:ui-monospace,Menlo,monospace;font-size:12px;opacity:.85;
flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
dialog{border:0;background:transparent;max-width:96vw;padding:0}
dialog img{max-width:96vw;max-height:96vh;border-radius:8px}
dialog::backdrop{background:rgba(0,0,0,.85)}"""

    sub = (f"{day} · {len(rows)} photos · {len(set(r['branch'] for r in rows))} branches · "
           f"{usable} usable"
           + (f" · {n_fail} download failures" if n_fail else ""))

    p = [f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
         f'<meta name="viewport" content="width=device-width,initial-scale=1">'
         f'<title>Gas box valve check — {day}</title><style>{css}</style></head><body>'
         f'<header><h1>Gas box valve check'
         f'{" — REVIEW COPY" if review_mode else ""}</h1><p>{sub}</p></header>'
         f'<div class="totals">']
    for k, l, col in SECTIONS:
        p.append(f'<div class="pill" style="background:{col}">{c.get(k,0)} &nbsp;'
                 f'{html.escape(l)}</div>')
    p.append('</div><div class="filters"><button class="on" data-f="all">All</button>'
             + "".join(f'<button data-f="{k}">{k.title()}</button>' for k,_,_ in SECTIONS)
             + '</div><div class="legend"><b>=</b> normal pass · <b>+</b> recovered by '
               'contrast pass (orange) · red = open valve · thin grey = 4th+ detection '
               'dropped by the top-3 rule.</div>')

    for k, l, col in SECTIONS:
        grp = [r for r in rows if r["verdict"] == k]
        if not grp: continue
        p.append(f'<section data-sec="{k}"><h2><span class="dot" style="background:{col}">'
                 f'</span>{html.escape(l)} ({len(grp)})</h2><div class="grid">')
        for r in grp:
            drp = (f'<div class="drp">dropped: {html.escape(r["dropped"])}</div>'
                   if r["dropped"] else "")
            fix = ""
            if review_mode:
                btns = "".join(
                    f'<button data-i="{r["idx"]}" data-v="{v}" '
                    f'style="{"background:"+COLOR[v]+";" if v==r["verdict"] else ""}" '
                    f'class="{"sel" if v==r["verdict"] else ""}">{v.title()}</button>'
                    for v in VERDICTS)
                fix = f'<div class="fix">{btns}</div>'
            corr = ""
            p.append(
                f'<div class="card" id="c{r["idx"]}">'
                f'<img src="data:image/jpeg;base64,{b64(r["img"])}" onclick="z(this.src)">'
                f'<div class="meta"><span class="tag" style="background:{col}">{k}</span>{corr}'
                f'<div class="b" style="margin-top:6px">{html.escape(r["brand"])} · '
                f'{html.escape(r["branch"])}</div>'
                f'<div class="sub">{html.escape(r["branch_name"][:52])}</div>'
                f'<div class="sub">{html.escape(r["user"][:40])}</div>'
                f'<div class="why">{html.escape(r["reason"])}</div>'
                f'<div class="sub">brightness {r["mean_lum"]} · {r["n_valves"]} valves · '
                f'{r["n_open"]} open</div>'
                f'<div class="det">{html.escape(r["detections"])}</div>{drp}{fix}</div></div>')
        p.append('</div></section>')

    if review_mode:
        p.append(f'''<div class="bar"><b id="n">0</b> corrections
<span class="out" id="o">(click a verdict button on any card you disagree with)</span>
<button onclick="cp()">Copy corrections</button></div>''')

    p.append('<dialog id="d" onclick="this.close()"><img id="di"></dialog><script>'
             'function z(s){document.getElementById("di").src=s;'
             'document.getElementById("d").showModal();}'
             'document.querySelectorAll(".filters button").forEach(b=>{b.onclick=()=>{'
             'document.querySelectorAll(".filters button").forEach(x=>'
             'x.classList.remove("on"));b.classList.add("on");const f=b.dataset.f;'
             'document.querySelectorAll("section").forEach(s=>{s.style.display='
             '(f==="all"||s.dataset.sec===f)?"":"none";});};});')
    if review_mode:
        p.append(f'''
const FP="{fp}"; const AUTO={json.dumps({r["idx"]: r["auto_verdict"] for r in rows})};
let O=JSON.parse(localStorage.getItem("gb_"+FP)||"{{}}");
const CO={json.dumps(COLOR)};
function paint(){{
 document.querySelectorAll(".fix button").forEach(b=>{{
  const i=b.dataset.i, cur=O[i]||AUTO[i];
  const on=b.dataset.v===cur;
  b.className=on?"sel":""; b.style.background=on?CO[b.dataset.v]:"#fafafa";
  b.style.color=on?"#fff":"#222";
 }});
 document.querySelectorAll(".card").forEach(c=>{{
  const i=c.id.slice(1); c.classList.toggle("ed", !!O[i]&&O[i]!==AUTO[i]);
 }});
 const ks=Object.keys(O).filter(i=>O[i]!==AUTO[i]);
 document.getElementById("n").textContent=ks.length;
 document.getElementById("o").textContent=ks.length?str():
   "(click a verdict button on any card you disagree with)";
 localStorage.setItem("gb_"+FP,JSON.stringify(O));
}}
function str(){{
 const ks=Object.keys(O).filter(i=>O[i]!==AUTO[i]).sort((a,b)=>a-b);
 return FP+":"+ks.map(i=>i+"="+O[i]).join(",");
}}
function cp(){{
 const s=str(); navigator.clipboard.writeText(s).then(()=>{{
  document.getElementById("o").textContent="copied: "+s;
 }},()=>{{prompt("Copy this:",s);}});
}}
document.querySelectorAll(".fix button").forEach(b=>{{b.onclick=()=>{{
 O[b.dataset.i]=b.dataset.v; paint();}};}});
paint();''')
    p.append('</script></body></html>')
    return "".join(p), c, usable


def build_pdf(rows, day, path, excluded=None, n_fail=0):
    """Plain A4 report: summary page, then two photo cards per page."""
    from fpdf import FPDF

    def T(x):
        """Built-in PDF fonts are latin-1 only — fold the typographic chars."""
        return (str(x).replace("\u2014", "-").replace("\u2013", "-")
                      .replace("\u2018", "'").replace("\u2019", "'")
                      .replace("\u201c", '"').replace("\u201d", '"')
                      .replace("\u00b7", "|").replace("\u2265", ">=")
                      .encode("latin-1", "replace").decode("latin-1"))

    c = Counter(r["verdict"] for r in rows)
    usable = len(rows) - c.get("UNUSABLE", 0)
    RGB = {"WRONG": (226, 75, 74), "REVIEW": (217, 130, 43),
           "UNUSABLE": (107, 107, 107), "correct": (29, 158, 117)}

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(False)
    L, W = 14, 182

    # ---- summary ----
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_xy(L, 18); pdf.cell(W, 9, T("Gas box valve check"), ln=1)
    pdf.set_font("Helvetica", "", 11); pdf.set_text_color(90, 90, 90)
    pdf.set_x(L)
    pdf.cell(W, 6, T(f"{day}   |   {len(rows)} photos   |   "
                     f"{len(set(r['branch'] for r in rows))} branches"), ln=1)
    pdf.set_text_color(0, 0, 0); pdf.ln(6)

    for k, label, _ in SECTIONS:
        pdf.set_x(L); pdf.set_fill_color(*RGB[k])
        pdf.set_text_color(255, 255, 255); pdf.set_font("Helvetica", "B", 12)
        pdf.cell(16, 9, T(f" {c.get(k,0)}"), fill=True)
        pdf.set_text_color(30, 30, 30); pdf.set_font("Helvetica", "", 11)
        pdf.cell(W - 16, 9, T(f"  {label}"), ln=1)
        pdf.ln(1.5)

    pdf.ln(4); pdf.set_font("Helvetica", "", 10); pdf.set_text_color(90, 90, 90)
    pdf.set_x(L)
    pdf.multi_cell(W, 5.5, T(f"{usable} of {len(rows)} photos actually show a gas box."))
    if excluded:
        eb = sorted({r["branch"] for r in excluded})
        pdf.set_x(L)
        pdf.multi_cell(W, 5.5, T(f"Not scored - 24-hour branches ({len(excluded)} photos): "
                                 + ", ".join(eb)))
    if n_fail:
        pdf.set_x(L); pdf.multi_cell(W, 5.5, T(f"{n_fail} photos failed to download."))
    pdf.set_text_color(0, 0, 0)

    flagged = [r for r in rows if r["verdict"] == "WRONG"]
    if flagged:
        pdf.ln(6); pdf.set_x(L); pdf.set_font("Helvetica", "B", 12)
        pdf.cell(W, 7, T("Branches flagged"), ln=1)
        pdf.set_font("Helvetica", "", 10)
        for r in flagged:
            pdf.set_x(L + 4)
            pdf.cell(W - 4, 6,
                     T(f"{r['brand']}  {r['branch']}   {r['branch_name'][:38]}   "
                       f"{r['user'][:26]}"), ln=1)

    # ---- photo cards, 2 per page ----
    CARD_H, IMG_H = 132, 96
    for k, label, _ in SECTIONS:
        grp = [r for r in rows if r["verdict"] == k]
        if not grp: continue
        pdf.add_page(); slot = 0
        pdf.set_x(L); pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(*RGB[k]); pdf.cell(W, 9, T(f"{label}  ({len(grp)})"), ln=1)
        pdf.set_text_color(0, 0, 0)
        top0 = pdf.get_y() + 2

        for r in grp:
            if slot == 2:
                pdf.add_page(); slot = 0; top0 = 16
            y = top0 + slot * CARD_H
            try:
                iw, ih = Image.open(r["img"]).size
                w = min(W, IMG_H * iw / ih)
                pdf.image(r["img"], x=L, y=y, h=IMG_H)
            except Exception:
                w = 0
            ty = y + IMG_H + 3
            pdf.set_xy(L, ty); pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(*RGB[k])
            pdf.cell(24, 5.5, T(k.upper()))
            pdf.set_text_color(0, 0, 0)
            pdf.cell(W - 24, 5.5, T(f"{r['brand']}  {r['branch']}"), ln=1)
            pdf.set_x(L); pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(95, 95, 95)
            pdf.cell(W, 4.6, T(f"{r['branch_name'][:46]}   {r['user'][:30]}"), ln=1)
            pdf.set_x(L)
            pdf.cell(W, 4.6, T(f"{r['reason']}   |   {r['n_valves']} valves, "
                               f"{r['n_open']} open   |   {r['detections'][:52]}"), ln=1)
            pdf.set_text_color(0, 0, 0)
            slot += 1

    pdf.output(path)
    return path


def apply_corrections(rows, s):
    if not s or not s.strip(): return 0, None
    s = s.strip()
    fp_given, _, body = s.partition(":")
    fp_now = None
    for r in rows: pass
    if not body: return 0, "corrections string has no body"
    idx = {r["idx"]: r for r in rows}
    n = 0
    for part in body.split(","):
        part = part.strip()
        if not part: continue
        k, _, v = part.partition("=")
        try: k = int(k)
        except ValueError: return n, f"bad index in '{part}'"
        if v not in VERDICTS: return n, f"unknown verdict '{v}'"
        if k not in idx: return n, f"index {k} not in this run"
        idx[k]["verdict"] = v
        idx[k]["corrected"] = True          # internal only, never shown
        idx[k]["reason"] = REASON[v]
        n += 1
    return n, None


def send_mail(subject, body, to, cc, attach_name, attach_bytes,
              maintype="text", subtype="html"):
    if not SMTP_PASS:
        print("!! SMTP_PASS not set — skipping send"); return
    m = EmailMessage()
    m["From"] = f"{SENDER_NAME} <{SMTP_USER}>"
    m["To"] = ", ".join(to)
    if cc: m["Cc"] = ", ".join(cc)
    m["Subject"] = subject
    m.set_content(body)
    m.add_attachment(attach_bytes, maintype=maintype, subtype=subtype,
                     filename=attach_name)
    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as s:
        s.login(SMTP_USER, SMTP_PASS)
        s.send_message(m)
    print(f"sent to {to} cc {cc}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["review", "send"])
    ap.add_argument("--corrections", default="")
    ap.add_argument("--date", default="")
    a = ap.parse_args()

    ksa = timezone(timedelta(hours=3))
    day = a.date or (datetime.now(ksa) - timedelta(days=DAYS_BACK)).strftime("%Y-%m-%d")
    print(f"stage={a.stage}  date={day}\n")

    shutil.rmtree(WORK, ignore_errors=True)
    for d in ("img", "enh", "out"): os.makedirs(f"{WORK}/{d}")

    records, excluded = collect(day)
    if not records:
        print("no gas box photos for this date — nothing to do"); return
    fp = fingerprint(records)
    print(f"\n{len(records)} photos  fingerprint={fp}")

    got, fails = download(records)
    print(f"downloaded {len(got)} | failed {len(fails)}")
    rows = score(got)

    if a.stage == "send" and a.corrections:
        given_fp = a.corrections.split(":", 1)[0].strip()
        if given_fp != fp:
            print(f"!! fingerprint mismatch: corrections are for '{given_fp}', "
                  f"this run is '{fp}'.")
            print("   The photo set changed since you reviewed. Re-run review "
                  "and redo the corrections.")
            sys.exit(1)
        n, err = apply_corrections(rows, a.corrections)
        if err: print(f"!! {err}"); sys.exit(1)
        print(f"applied {n} corrections")

    review = a.stage == "review"
    doc, c, usable = build_html(rows, day, fp, review, len(fails))
    if review:
        out = f"{WORK}/valve_check_{day}.html"
        open(out, "w", encoding="utf-8").write(doc)
        print(f"review html {os.path.getsize(out)/1e6:.1f} MB")

    csvp = f"{WORK}/valve_check_{day}.csv"
    with open(csvp, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=[k for k in rows[0] if k != "img"])
        w.writeheader()
        for r in rows: w.writerow({k: v for k, v in r.items() if k != "img"})

    print(f"\n{c.get('correct',0)} correct | {c.get('WRONG',0)} WRONG | "
          f"{c.get('REVIEW',0)} review | {c.get('UNUSABLE',0)} unusable")

    flagged = [r for r in rows if r["verdict"] == "WRONG"]
    lines = [f"Gas box valve check — {day}", "",
             f"{len(rows)} photos across {len(set(r['branch'] for r in rows))} branches",
             f"  {c.get('correct',0)} all three closed",
             f"  {c.get('WRONG',0)} open valve detected",
             f"  {c.get('REVIEW',0)} model unsure — human check",
             f"  {c.get('UNUSABLE',0)} photo does not show the gas box", ""]
    if excluded:
        eb = sorted({r["branch"] for r in excluded})
        lines += [f"Not scored ({len(excluded)} photos from 24-hour branches): "
                  + ", ".join(eb), ""]
    if flagged:
        lines += ["Branches flagged:"] + \
                 [f"  {r['brand']} {r['branch']} — {r['branch_name'][:40]} ({r['user']})"
                  for r in flagged] + [""]
    lines += ["Open the attached HTML for the annotated photos."]

    if review:
        lines = ([f"REVIEW COPY — fingerprint {fp}", "",
                  "Open the attachment, correct any verdict you disagree with, then",
                  "run the 'Gas box valve check — send' workflow and paste the",
                  "corrections string. Paste nothing to send as-is.", ""] + lines)
        send_mail(f"[REVIEW] Gas box valve check — {day}", "\n".join(lines),
                  MAIL_REVIEWER or [SMTP_USER], [],
                  f"valve_check_{day}_REVIEW.html", doc.encode())
    else:
        pdfp = f"{WORK}/valve_check_{day}.pdf"
        build_pdf(rows, day, pdfp, excluded, len(fails))
        print(f"pdf {os.path.getsize(pdfp)/1e6:.1f} MB")
        lines[-1] = "The attached PDF has the annotated photos."
        send_mail(f"Gas box valve check — {day}", "\n".join(lines),
                  MAIL_TO, MAIL_CC, f"valve_check_{day}.pdf",
                  open(pdfp, "rb").read(), "application", "pdf")


if __name__ == "__main__":
    main()
