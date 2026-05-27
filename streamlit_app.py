"""
STOCKHOLMS STADSMISSION · Stiftelse-matchning
Simplified Streamlit prototype mirroring the FastHTML mockup.

Phase 1 (Ingest):  JSON → Polars → Postgres(stiftelser) → chunk+embed (e5) → pgvector/HNSW
Phase 2 (Match):   programme profile → embed query → ANN → group-by → LLM rerank → fit/gap → draft

Run:
    pip install streamlit
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import math
import random
import re
import time

import streamlit as st

# =============================================================================
# PAGE CONFIG + BRANDING / READABILITY-FIRST CSS
# =============================================================================
st.set_page_config(
    page_title="Stadsmission · Stiftelse-matchning",
    page_icon="🟧",
    layout="wide",
    initial_sidebar_state="expanded",
)

ORANGE = "#F07E1A"

st.markdown(
    f"""
    <style>
      /* Stadsmission brand mark */
      .brand-mark {{
        background: {ORANGE};
        color: #000;
        font-family: 'Arial Black', sans-serif;
        font-weight: 900;
        padding: 8px 12px 6px;
        line-height: .85;
        display: inline-block;
        letter-spacing: -.01em;
      }}
      .brand-mark div {{ display: block; color: #000; }}

      h1, h2, h3 {{ letter-spacing: -.01em; }}
      .accent {{ color: {ORANGE}; font-weight: 600; }}
      .mono {{ font-family: ui-monospace, 'JetBrains Mono', Menlo, monospace; }}

      /* === Foundation card — explicit colors so it reads in light & dark themes === */
      .fcard {{
        border: 1px solid #d8d8d8;
        border-left: 4px solid {ORANGE};
        padding: 16px 18px;
        margin-bottom: 18px;
        background: #fafafa;
        color: #1a1a1a;
      }}
      .fcard, .fcard * {{ color: #1a1a1a; }}
      .fcard h4 {{ margin: 0 0 6px; color: #1a1a1a !important; font-size: 16px; font-weight: 700; }}
      .fcard .sec-h {{
        font-size: 11px; text-transform: uppercase; letter-spacing: .14em;
        color: #666 !important; font-weight: 700; margin: 14px 0 6px;
      }}
      .badge {{
        display: inline-block; padding: 1px 7px; border: 1px solid #ccc;
        font-family: ui-monospace, monospace; font-size: 11px;
        margin: 0 4px 4px 0; background: #fff; color: #444 !important;
      }}
      .badge.lan {{ border-color: {ORANGE}; color: {ORANGE} !important; }}
      .andamal {{
        font-family: ui-monospace, monospace; font-size: 12px;
        background: #fff; border-left: 2px solid {ORANGE};
        padding: 10px 12px; color: #1a1a1a !important;
        line-height: 1.55; white-space: pre-wrap;
      }}
      .andamal mark {{ background: rgba(240,126,26,.25); color: #6a3500 !important; padding: 0 2px; }}
      .notes {{ font-size: 11px; color: #777 !important; margin-top: 6px; font-style: italic; }}
      .draft {{
        font-family: Georgia, serif; font-size: 14px; line-height: 1.65;
        background: #fff; border-left: 2px solid {ORANGE};
        padding: 14px 16px; color: #1a1a1a !important;
      }}
      .draft .salu {{ color: {ORANGE} !important; font-weight: 700; }}
      .draft em {{ color: #555 !important; font-style: italic; }}
      .draft-meta {{ margin-top: 6px; font-family: ui-monospace, monospace; font-size: 10px; color: #777 !important; }}
      .apptype {{
        background: rgba(122,170,255,.12); border: 1px solid #b8ccea;
        padding: 10px 12px; color: #1a1a1a !important; font-size: 13px;
      }}
      .apptype b {{ color: #1a3a6a !important; }}
      .fitgap {{ display: flex; gap: 10px; margin-top: 6px; flex-wrap: wrap; }}
      .fitgap > div {{
        flex: 1 1 200px; padding: 10px 12px; background: #fff;
        border: 1px solid #ddd; min-width: 0;
      }}
      .fitgap .hh {{ font-size: 10px; font-weight: 700; text-transform: uppercase;
        letter-spacing: .14em; margin-bottom: 6px; }}
      .fitgap .aligned .hh {{ color: #2e7d32 !important; }}
      .fitgap .gap .hh     {{ color: #b6841a !important; }}
      .fitgap .reframe .hh {{ color: #1a5dbb !important; }}
      .fitgap ul {{ margin: 0; padding-left: 16px; font-size: 12px; line-height: 1.5; color: #333 !important; }}
      .fitgap ul li {{ margin-bottom: 4px; color: #333 !important; }}
      .scorebox {{
        font-family: ui-monospace, monospace; font-size: 11px;
        color: #666 !important; margin-top: 4px;
      }}
      .scorebox b {{ color: {ORANGE} !important; font-weight: 700; }}
      .rank-badge {{
        background: {ORANGE}; color: #000 !important;
        font-family: 'Arial Black', sans-serif; font-size: 22px;
        width: 54px; min-width: 54px; height: 54px;
        display: flex; align-items: center; justify-content: center;
        font-weight: 900; flex-shrink: 0;
      }}
      .v2 {{
        border: 1px dashed #bbb; padding: 8px 10px;
        font-family: ui-monospace, monospace; font-size: 11px;
        color: #888 !important; background: repeating-linear-gradient(45deg, transparent 0 6px, rgba(0,0,0,.02) 6px 12px);
      }}
      .v2::before {{ content: 'v2'; background: {ORANGE}; color: #000;
        padding: 1px 5px; font-weight: 700; margin-right: 8px; }}

      /* Pipeline strip */
      .pnode {{ text-align: center; padding: 10px 6px; border: 1px solid #ccc;
        background: #fff; font-size: 12px; color: #1a1a1a; }}
      .pnode.active {{ border-color: {ORANGE}; background: rgba(240,126,26,.08); }}
      .pnode.done {{ border-color: #2e7d32; background: rgba(46,125,50,.06); }}
      .pnode .nm {{ font-size: 10px; letter-spacing: .14em; text-transform: uppercase;
        color: #666 !important; font-weight: 700; }}
      .pnode.active .nm {{ color: {ORANGE} !important; }}
      .pnode.done .nm {{ color: #2e7d32 !important; }}
      .pnode .ic {{ font-size: 18px; margin: 6px 0 4px; }}
      .pnode .mt {{ font-family: ui-monospace, monospace; font-size: 10px;
        color: #777 !important; min-height: 12px; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# HEADER
# =============================================================================
hdr_l, hdr_r = st.columns([1, 3])
with hdr_l:
    st.markdown(
        '<div class="brand-mark"><div>STOCKHOLMS</div><div>STADSMISSION</div></div>',
        unsafe_allow_html=True,
    )
with hdr_r:
    st.markdown(
        "### Stiftelse-matchning · prototype  \n"
        '<span style="color:#888;font-family:ui-monospace,monospace;font-size:12px">'
        'Phase 1 (Ingest) + Phase 2 (Match &amp; Draft) · Postgres + pgvector + '
        'e5-multilingual-base · DEMO MODE</span>',
        unsafe_allow_html=True,
    )

st.divider()

# =============================================================================
# DATA — same source-of-truth as the FastHTML mockup
# =============================================================================

PROGRAMMES = {
    "hemloshet": {
        "label": "Hemlöshet",
        "text": (
            "Stockholms Stadsmission driver härbärgen, lågtröskelboenden och dagverksamheter "
            "för människor i hemlöshet i Stockholms län. Vi möter vuxna i akut bostadslöshet, "
            "långvarig hemlöshet och utsatta EU-medborgare. Insatserna omfattar nattlig "
            "övernattning, hygien, måltider, social rådgivning, samt motiverande samtal mot "
            "stadigvarande boende. Vi söker bidrag för att utöka antal platser, vinterprotokoll "
            "vid kyla under -5°C, samt riktade insatser för kvinnor och hbtqi-personer."
        ),
    },
    "barn_unga": {
        "label": "Barn och unga",
        "text": (
            "Verksamheten omfattar öppna mötesplatser för barn och unga i socioekonomiskt "
            "utsatta områden i Stockholm, läxhjälp, sociala fritidsaktiviteter samt riktat stöd "
            "till barn i familjer med försörjningsstöd, missbruk eller psykisk ohälsa. Vi arbetar "
            "med vård, fostran och utbildning enligt FN:s barnkonvention. Bidrag söks för "
            "lokalkostnader, ledarutbildning och utrustning till sommarkollo."
        ),
    },
    "arbetsint": {
        "label": "Arbetsintegration",
        "text": (
            "Stadsmissionens arbetsintegration vänder sig till personer långt från arbetsmarknaden "
            "— långtidsarbetslösa, nyanlända, personer i missbruksrehabilitering och personer med "
            "psykisk ohälsa. Verksamheten består av språkkafé i svenska, arbetspraktik i våra "
            "sociala företag, individuell coachning och samverkan med Arbetsförmedlingen. Vi söker "
            "medel för handledartjänster och språkmaterial."
        ),
    },
    "secondhand": {
        "label": "Second hand",
        "text": (
            "Stadsmissionens second hand-butiker säljer skänkta varor och erbjuder arbetsträning "
            "för personer som står långt från arbetsmarknaden. Överskottet finansierar våra "
            "sociala verksamheter. Behovet är investering i ny butik i Solna samt ombyggnad av "
            "sorteringslager — bidrag söks för inventarier, hyllor och transportbil."
        ),
    },
    "matbutik": {
        "label": "Sociala matbutiker (Matmissionen)",
        "text": (
            "Matmissionen är vår sociala matbutik som säljer livsmedel till kraftigt reducerade "
            "priser till låginkomsthushåll i Stockholm. Vi tar emot matsvinn från livsmedelskedjor "
            "och minskar både matsvinn och matfattigdom. Bidrag söks för kylkedjeutrustning, "
            "transportbil samt utökning till en andra butik i söderort."
        ),
    },
    "akut_utsatthet": {
        "label": "Stöd vid akut utsatthet",
        "text": (
            "Verksamheten ger krisstöd till människor i akut social nöd: våldsutsatta kvinnor, "
            "utsatta EU-medborgare, personer i hemlöshet vid utskrivning från slutenvård. Vi "
            "erbjuder kortvarigt boende, mat, hygien, läkar- och tandläkarkontakter, samt "
            "rådgivning. Bidrag söks för akut-fond, jourtelefon och utvidgad uppsökande "
            "verksamhet under vintern."
        ),
    },
}

# Real foundations sampled from stiftelser_2026-05-26_1154.json
STIFTELSER = [
    dict(
        id=1003001, namn="Olle Engkvists Stiftelse", orgnr="802005-1467", ort="STOCKHOLM",
        lankod=1, kommunkod=180, typ=6, size="stor",
        andamal=(
            "Att främja vård eller uppfostran av barn, vård av behövande ålderstigna, sjuka eller "
            "lytta och vetenskaplig forskning."
        ),
        highlight=["vård eller uppfostran av barn", "behövande ålderstigna, sjuka eller lytta"],
        notes="En av de större allmännyttiga stiftelserna · breda anslag inom social omsorg.",
        rel=dict(hemloshet=.78, barn_unga=.91, arbetsint=.48, secondhand=.32, matbutik=.39, akut_utsatthet=.82),
    ),
    dict(
        id=1011207, namn="Stiftelsen Clas Groschinskys Minnesfond", orgnr="802002-7848", ort="STOCKHOLM",
        lankod=1, kommunkod=180, typ=6, size="medel",
        andamal=(
            "Att främja vård och uppfostran av barn och lämna understöd för beredande av barns och "
            "ungdoms undervisning och utbildning samt att utöva hjälpverksamhet bland behövande "
            "ålderstigna, sjuka och lytta."
        ),
        highlight=["vård och uppfostran av barn", "hjälpverksamhet bland behövande"],
        notes="Etablerade utlysningar varje vår.",
        rel=dict(hemloshet=.69, barn_unga=.94, arbetsint=.51, secondhand=.28, matbutik=.41, akut_utsatthet=.77),
    ),
    dict(
        id=1004409, namn="Sven och Dagmar Saléns Stiftelse", orgnr="802005-7415", ort="STOCKHOLM",
        lankod=1, kommunkod=180, typ=6, size="medel-stor",
        andamal=(
            "Stiftelsens ändamål är att främja barns eller ungdoms vård och fostran eller "
            "utbildning; vård av behövande ålderstigna, sjuka eller lytta; samt vetenskaplig "
            "undervisning eller forskning."
        ),
        highlight=["barns eller ungdoms vård och fostran", "behövande ålderstigna, sjuka"],
        notes="Tre-delat ändamål · social omsorg + forskning.",
        rel=dict(hemloshet=.66, barn_unga=.90, arbetsint=.46, secondhand=.30, matbutik=.38, akut_utsatthet=.72),
    ),
    dict(
        id=1002218, namn="Svenska Stiftelsen för Frälsningsarmén", orgnr="802005-1051", ort="STOCKHOLM",
        lankod=1, kommunkod=180, typ=6, size="stor",
        andamal=(
            "Stiftelsen har till ändamål att för främjandet av Frälsningsarméns verksamhet för "
            "barns och ungdoms vård, fostran och utbildning eller för att främja vård av behövande "
            "ålderstigna, sjuka eller lytta."
        ),
        highlight=["barns och ungdoms vård, fostran", "behövande ålderstigna, sjuka"],
        notes="Knuten till Frälsningsarmén — kan begränsa extern utdelning.",
        rel=dict(hemloshet=.88, barn_unga=.80, arbetsint=.62, secondhand=.45, matbutik=.58, akut_utsatthet=.89),
    ),
    dict(
        id=1007711, namn="Anders Sandrews Stiftelse", orgnr="802005-3315", ort="STOCKHOLM",
        lankod=1, kommunkod=180, typ=6, size="stor",
        andamal=(
            "Ändamålet skall vara att, utan begränsning till viss familj, vissa familjer eller "
            "bestämda personer, dels främja vård och uppfostran av barn, som saknar erforderliga "
            "medel, dels lämna understöd i form av exempelvis stipendier."
        ),
        highlight=["vård och uppfostran av barn, som saknar erforderliga medel"],
        notes='Tydlig socioekonomisk skrivning — "saknar erforderliga medel".',
        rel=dict(hemloshet=.70, barn_unga=.93, arbetsint=.49, secondhand=.30, matbutik=.42, akut_utsatthet=.78),
    ),
    dict(
        id=1009023, namn="Stiftelsen Ragnhild och Einar Lundströms Minne", orgnr="802005-2655", ort="STOCKHOLM",
        lankod=1, kommunkod=180, typ=6, size="medel",
        andamal=(
            "Stiftelsens ändamål är att utdela understöd, anslag, stipendier till främjande av "
            "barns och ungdoms vård och fostran eller utbildning, ävensom vård av behövande "
            "ålderstigna, sjuka eller lytta."
        ),
        highlight=["barns och ungdoms vård och fostran", "behövande ålderstigna"],
        notes="Klassisk bredspektra-formulering.",
        rel=dict(hemloshet=.71, barn_unga=.88, arbetsint=.55, secondhand=.33, matbutik=.42, akut_utsatthet=.79),
    ),
    dict(
        id=1014502, namn="Thora och John Nilssons Stiftelse", orgnr="802004-9818", ort="STOCKHOLM",
        lankod=1, kommunkod=180, typ=6, size="liten-medel",
        andamal=(
            "Främja vård av behövande ålderstigna, sjuka eller lytta inom S:t Görans församling och "
            "dels främja barns och ungdoms vård och fostran eller utbildning, företrädesvis "
            "alkohol- och narkotikamissbrukare."
        ),
        highlight=["behövande ålderstigna", "företrädesvis alkohol- och narkotikamissbrukare"],
        notes="⚠ Geo-begränsat (S:t Göran) · explicit missbruksinriktning.",
        rel=dict(hemloshet=.86, barn_unga=.75, arbetsint=.80, secondhand=.41, matbutik=.50, akut_utsatthet=.85),
    ),
    dict(
        id=1015820, namn="Byggnadsstiftelsen S:t Erik", orgnr="802005-9882", ort="STOCKHOLM",
        lankod=1, kommunkod=180, typ=6, size="stor",
        andamal=(
            "Stiftelsen skall med avkastningen av sina tillgångar främja sådan social verksamhet, "
            "som avser vård och uppfostran av barn, beredande av undervisning, vård av behövande "
            "ålderstigna, sjuka eller lytta, samt arbetshjälp åt behövande."
        ),
        highlight=["vård och uppfostran av barn", "arbetshjälp åt behövande"],
        notes='Inkluderar "arbetshjälp" — bäring mot arbetsintegration.',
        rel=dict(hemloshet=.74, barn_unga=.83, arbetsint=.88, secondhand=.58, matbutik=.49, akut_utsatthet=.81),
    ),
    dict(
        id=1018330, namn="Stiftelsen Maja & J.P Åhlén", orgnr="802005-7126", ort="STOCKHOLM",
        lankod=1, kommunkod=180, typ=6, size="medel",
        andamal=(
            "Stiftelsen har till ändamål att främja barn eller ungdoms vård och fostran eller "
            "utbildning, vård av behövande ålderstigna, sjuka eller lytta, vetenskaplig "
            "undervisning eller forskning."
        ),
        highlight=["barn eller ungdoms vård", "behövande ålderstigna"],
        notes="Klassisk skrivning · forskning som tredje ändamål.",
        rel=dict(hemloshet=.65, barn_unga=.87, arbetsint=.47, secondhand=.29, matbutik=.39, akut_utsatthet=.72),
    ),
    dict(
        id=1020014, namn="Madinah insamlingsstiftelse", orgnr="802480-9920", ort="KISTA",
        lankod=1, kommunkod=180, typ=6, size="liten-medel",
        andamal=(
            "Stiftelsens första ändamål är att bedriva sådan verksamhet som främjar tillvaron i "
            "utsatta områden i Sverige, genom att verka för bättring vad gäller såväl den sociala "
            "som materiella situationen."
        ),
        highlight=["utsatta områden i Sverige", "sociala som materiella situationen"],
        notes="Insamlingsstiftelse · konfessionell profil.",
        rel=dict(hemloshet=.79, barn_unga=.74, arbetsint=.71, secondhand=.55, matbutik=.68, akut_utsatthet=.82),
    ),
    dict(
        id=1022115, namn="Stiftelsen Sufi Service Committee", orgnr="802482-1107", ort="TYRESÖ",
        lankod=1, kommunkod=138, typ=6, size="liten",
        andamal=(
            "Att främja tillvaron i utsatta områden i Sverige och inom humanitär hjälp i Afrikas "
            "mest utsatta länder, genom att verka för bättring vad gäller såväl den sociala som "
            "materiella situationen — vård, fostran och utbildning av barn och ungdom, samt vård "
            "av behövande ålderstigna, sjuka och hemlösa."
        ),
        highlight=["utsatta områden", "hemlösa"],
        notes="Bred social skrivning · explicit nämner hemlösa.",
        rel=dict(hemloshet=.93, barn_unga=.71, arbetsint=.66, secondhand=.40, matbutik=.61, akut_utsatthet=.92),
    ),
    dict(
        id=1025603, namn="Läkarmissionen stiftelse för filantropisk verksamhet", orgnr="802005-9989",
        ort="VÄLLINGBY", lankod=1, kommunkod=180, typ=6, size="stor",
        andamal=(
            "Stiftelsens huvudändamål är att främja kristen läkarmission, vilket innebär "
            "internationell hjälpverksamhet, speciellt för främjande av barns och ungdoms vård, "
            "fostran och utbildning, samt hjälp till behövande."
        ),
        highlight=["barns och ungdoms vård", "behövande"],
        notes="⚠ Främst internationell hjälp · svensk verksamhet sekundär.",
        rel=dict(hemloshet=.55, barn_unga=.82, arbetsint=.38, secondhand=.24, matbutik=.43, akut_utsatthet=.65),
    ),
]

# Geo / typ decoders ("decoded at PULL, not at ingest" — astute.md #9)
LAN = {1: "Stockholms län", 3: "Uppsala län", 4: "Södermanlands län", 5: "Östergötlands län"}
TYP = {6: "Allmännyttig stiftelse", 3: "Insamlingsstiftelse", -1: "(okänd)"}
KOMMUN = {180: "Stockholm", 138: "Tyresö", 181: "Södertälje", 188: "Norrtälje"}


# =============================================================================
# CORE LOGIC — pluggable seams for the real backend
# =============================================================================

def score_for(s: dict, prog: str, agg: str, rerank: bool) -> float:
    """SEAM: replace with embed(prog) → ANN over chunks → group-by stiftelse_id."""
    base = s["rel"].get(prog, 0.4)
    if agg == "max":
        base = min(0.98, base + 0.03)
    elif agg == "sum":
        base = min(0.99, base + 0.06)
    if rerank:
        base = min(0.99, base + 0.04)
    base += math.sin(s["id"] * 0.0001) * 0.012
    return max(0.05, min(0.99, base))


def highlight_andamal(text: str, terms: list[str]) -> str:
    h = text
    for t in terms:
        pat = re.compile(re.escape(t), re.IGNORECASE)
        h = pat.sub(lambda m: f"<mark>{m.group(0)}</mark>", h)
    return h


def compose_fit_gap(prog: str, f: dict) -> dict:
    """SEAM: replace with 1 LLM call per foundation, same return shape."""
    aligned: list[str] = []
    andamal = f["andamal"]
    if prog == "barn_unga" and any(re.search(r"barn|ungdom", h, re.I) for h in f["highlight"]):
        aligned.append("Explicit nämner barns vård/fostran — exakt matchning mot programmet.")
    if prog in ("hemloshet", "akut_utsatthet") and any(
        re.search(r"hemlös|ålderstigna|behövande", h, re.I) for h in f["highlight"]
    ):
        aligned.append('ANDAMAL inkluderar "behövande"/"hemlösa" — bäring direkt mot målgrupp.')
    if re.search(r"arbetshjälp|missbruk", andamal, re.I) and prog in ("arbetsint", "hemloshet"):
        aligned.append("ANDAMAL nämner arbetshjälp/missbruk — passar arbetsintegration.")
    if f["lankod"] == 1:
        aligned.append("Geo: Stockholms län — överensstämmer med verksamheten.")
    if f["typ"] == 6:
        aligned.append("TYP 6 (allmännyttig) — öppen utdelningsstruktur.")
    while len(aligned) < 3:
        aligned.append("Brett socialt ändamål · rymmer Stadsmissionens kärnverksamhet.")

    gaps: list[str] = []
    if re.search(r"församling|inom\s+S:t", andamal, re.I):
        gaps.append("Geo-begränsning till specifik församling — kontrollera om verksamheten täcks.")
    if re.search(r"Frälsningsarmén|kristen|läkarmission", andamal, re.I):
        gaps.append("Konfessionell/organisatorisk knytning — extern utdelning kan vara begränsad.")
    if re.search(r"internationell|utland|Afrikas", andamal, re.I):
        gaps.append("Internationellt fokus — svensk verksamhet kan vara sekundär.")
    if re.search(r"forskning|vetenskaplig", andamal, re.I):
        gaps.append("Forskning som parallellt ändamål — risk för konkurrens om medlen.")
    if len(gaps) < 2:
        gaps.append("Stor stiftelse → många sökande, konkurrensen om medlen är hög.")

    reframes = {
        "hemloshet":      'Rama in ansökan som "vård av behövande ålderstigna" om foundation har äldre-fokus — många härbärgesgäster är 55+.',
        "barn_unga":      'Vinkla mot "barns fostran och utbildning" — våra mötesplatser har dokumenterad pedagogisk komponent.',
        "arbetsint":      'Lyft "arbetshjälp åt behövande" som direkt citat ur ANDAMAL i ansökans öppningsstycke.',
        "secondhand":     'Beskriv second hand som arbetsträning för "behövande", inte som butiksverksamhet.',
        "matbutik":       'Positionera Matmissionen som matfattigdomsinsats för "behövande", inte som handel.',
        "akut_utsatthet": 'Använd formuleringen "akut social nöd" som matchar äldre stiftelseformuleringar om "nödställda".',
    }
    return {"aligned": aligned[:3], "gaps": gaps[:2], "reframe": [reframes[prog]]}


def suggest_project_type(prog: str, f: dict) -> str:
    m = {
        "hemloshet":      'Riktat projekt: "Vinterplatser för kvinnor 2026" — 12 mån, ~450 kSEK, mätbart utfall i platser & beläggning.',
        "barn_unga":      'Projektidé: "Läxhjälp + mötesplats i Skärholmen" — 18 mån, ~620 kSEK, samverkan med skolor.',
        "arbetsint":      'Pilot: "Språkpraktik i Matmissionen" — 12 mån, 8 praktikplatser, ~380 kSEK i handledartid.',
        "secondhand":     'Engångsbidrag: "Sorteringslager Solna" — engångsstöd, ~250 kSEK, möbler & hyllor.',
        "matbutik":       'Investeringsbidrag: "Andra Matmissionen-butik söderort" — engångsstöd, ~900 kSEK, lokal & kyl.',
        "akut_utsatthet": 'Verksamhetsstöd: "Akut-fond EU-medborgare vinter 2026/27" — säsongstöd, ~280 kSEK.',
    }
    return m.get(prog, "Verksamhetsbidrag enligt foundation-storlek.")


def compose_draft(prog: str, f: dict) -> str:
    """SEAM: replace with LLM call returning ~200 words, same HTML wrapping."""
    p = PROGRAMMES[prog]["label"]
    intros = {
        "hemloshet":      "Stockholms Stadsmission ansöker härmed om bidrag till vår verksamhet inom hemlöshet i Stockholms län.",
        "barn_unga":      "Stockholms Stadsmission ansöker om medel till våra öppna mötesplatser och stödinsatser för barn och unga i utsatta livssituationer.",
        "arbetsint":      "Stockholms Stadsmission ansöker om bidrag till vår arbetsintegrerande verksamhet för personer långt från arbetsmarknaden.",
        "secondhand":     "Stockholms Stadsmission ansöker om medel till våra second hand-butiker, som finansierar social verksamhet och erbjuder arbetsträning.",
        "matbutik":       "Stockholms Stadsmission ansöker om bidrag till Matmissionen, vår sociala matbutik som motverkar matsvinn och matfattigdom.",
        "akut_utsatthet": "Stockholms Stadsmission ansöker om medel till vårt akuta stöd för människor i social nöd — utsatta EU-medborgare, våldsutsatta och hemlösa.",
    }
    closes = {
        "hemloshet":      "Bidraget skulle möjliggöra utökade härbärgesplatser samt vinterprotokoll under perioder med temperaturer under -5°C.",
        "barn_unga":      "Vi söker medel till lokalkostnader, ledarutbildning och sommarkollo för barn ur familjer med försörjningsstöd.",
        "arbetsint":      "Bidraget används till handledartjänster och språkmaterial i våra sociala arbetsintegrationsföretag.",
        "secondhand":     "Medlen skulle finansiera ombyggnad av sorteringslager och inventarier till ny butik i Solna.",
        "matbutik":       "Bidraget skulle finansiera kylkedjeutrustning och möjliggöra öppnandet av en andra Matmissionen-butik i söderort.",
        "akut_utsatthet": "Medlen används till akut-fond, jourtelefon och utökad uppsökande verksamhet under vintern.",
    }
    highlight_quotes = " och ".join('"' + h + '"' for h in f["highlight"])
    bridge = (
        f"Stiftelsens ändamål — {highlight_quotes} — har direkt bäring på vår verksamhet "
        f"inom {p.lower()}, som omfattar mer än 4 000 personer årligen i Stockholms län."
    )
    body = (
        "Stockholms Stadsmission är en idéburen organisation som sedan 1853 arbetar med social "
        "utsatthet i Stockholm. Vår verksamhet drivs i nära samverkan med Stockholms stad, "
        "Region Stockholm och privata givare. Senaste verksamhetsåret nådde vi 23 800 unika "
        "personer genom våra härbärgen, mötesplatser, matbutiker och arbetsintegration."
    )
    sign = (
        "Vi bifogar gärna ekonomisk redovisning, verksamhetsberättelse och referenser. "
        "Med vänlig hälsning,<br/><em>— [Verksamhetschef · Stockholms Stadsmission]</em>"
    )
    return (
        '<span class="salu">Till styrelsen för ' + f["namn"] + ',</span><br/><br/>'
        + intros[prog] + '<br/><br/>'
        + bridge + '<br/><br/>'
        + body + '<br/><br/>'
        + closes[prog] + '<br/><br/>'
        + sign
    )


# =============================================================================
# PIPELINE STRIP RENDERER
# =============================================================================

PIPE_STAGES = [
    ("json",     "JSON-källa",     "{ }"),
    ("polars",   "Polars frame",   "▤"),
    ("postgres", "stiftelser",     "▦"),
    ("chunk",    "Chunk + embed",  "≋"),
    ("chunks",   "chunks (HNSW)",  "◈"),
]


def render_pipeline(states):
    cols = st.columns([2, 1, 2, 1, 2, 1, 2, 1, 2])
    for i, (key, name, icon) in enumerate(PIPE_STAGES):
        col = cols[i * 2]
        status, meta = states.get(key, ("", ""))
        cls = ("pnode " + status).strip()
        col.markdown(
            '<div class="' + cls + '">'
            '<div class="nm">' + name + '</div>'
            '<div class="ic">' + icon + '</div>'
            '<div class="mt">' + (meta or "") + '</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        if i < len(PIPE_STAGES) - 1:
            arrow = cols[i * 2 + 1]
            arrow_color = ORANGE if status == "active" else "#aaa"
            arrow.markdown(
                '<div style="text-align:center;padding-top:32px;color:' + arrow_color + '">→</div>',
                unsafe_allow_html=True,
            )


# =============================================================================
# SESSION STATE
# =============================================================================
if "ingest_done" not in st.session_state:
    st.session_state.ingest_done = False
if "ingest_log" not in st.session_state:
    st.session_state.ingest_log = []
if "ingest_stats" not in st.session_state:
    st.session_state.ingest_stats = {"rows": 0, "chunks": 0, "vecs": 0, "dim": 768}


# =============================================================================
# TABS
# =============================================================================
tab1, tab2 = st.tabs([
    "**01 · Ingest**  (JSON → Postgres → pgvector)",
    "**02 · Match & Draft**  (programme → top-N → utkast)",
])

# -----------------------------------------------------------------------------
# TAB 1 · INGEST
# -----------------------------------------------------------------------------
with tab1:
    left, right = st.columns([1, 2.4], gap="large")

    with left:
        st.markdown("##### Pipeline-konfiguration")
        with st.form("ingest_form", border=False):
            source = st.selectbox(
                "Källa",
                options=[
                    "stiftelser_2026-05-26_1154.json (17 476 rader)",
                    "stiftelser_2026-04-12_0900.json (17 421 rader)",
                    "v2 · scraper @ lansstyrelsen.se",
                ],
                index=0,
            )
            text_col = st.selectbox(
                "Text-fält att embedda",
                options=["andamal", "namn", "firmateckning"],
                index=0,
                help="ANDAMAL är det semantiskt rika fältet — rekommenderat.",
            )
            chunk_strat = st.radio(
                "Chunk-strategi",
                options=["paragraph", "sentence", "window"],
                index=0,
                horizontal=True,
                help="Paragraf-först, mening som fallback när chunk > max_chars.",
            )
            max_chars = st.slider("max_chars per chunk", 200, 1200, 600, 50)
            model = st.selectbox(
                "Embedding-modell",
                options=[
                    "intfloat/multilingual-e5-base · 768d",
                    "intfloat/multilingual-e5-large · 1024d",
                    "KB/kb-bert-base-swedish-cased · 768d",
                ],
                index=0,
                help='e5 → "passage: " prefix vid ingest, "query: " vid sök',
            )
            batch_size = st.slider("Batch size", 8, 256, 64, 8)

            st.markdown("**Steg**")
            normalize = st.toggle("Normalize embeddings", value=True)
            hnsw = st.toggle("HNSW index (cosine)", value=True)
            enrich = st.toggle("v2 · table-enricher (decode + scrape + klassificera)", value=False)
            verify = st.toggle("Verify (kör test-query)", value=True)

            submitted = st.form_submit_button(
                "▶ Kör ingest-pipeline", type="primary", use_container_width=True
            )

        if st.button("Reset", use_container_width=True):
            st.session_state.ingest_done = False
            st.session_state.ingest_log = []
            st.session_state.ingest_stats = {"rows": 0, "chunks": 0, "vecs": 0, "dim": 768}
            st.rerun()

    with right:
        st.markdown("##### Pipeline-telemetri")

        pipeline_box = st.empty()
        stats_box = st.empty()
        progress_box = st.empty()
        log_box = st.empty()

        def render_stats():
            s = st.session_state.ingest_stats
            with stats_box.container():
                c1, c2, c3, c4 = st.columns(4)
                total_rows = 17476
                c1.metric(
                    "stiftelser-rader",
                    f"{s['rows']:,}".replace(",", " "),
                    f"av {total_rows:,}".replace(",", " "),
                )
                c2.metric(
                    "chunks",
                    f"{s['chunks']:,}".replace(",", " "),
                    f"~{(s['chunks']/max(1,s['rows']*0.987)):.2f}/rad" if s["rows"] else None,
                )
                c3.metric(
                    "embeddings",
                    f"{s['vecs']:,}".replace(",", " "),
                    f"{s['dim']}d",
                )
                c4.metric(
                    "HNSW",
                    "BUILT" if (st.session_state.ingest_done and hnsw) else "—",
                    "cosine · m=16" if (st.session_state.ingest_done and hnsw) else None,
                )

        def render_log():
            with log_box.container():
                st.markdown("**Stream-logg**")
                if st.session_state.ingest_log:
                    st.code("\n".join(st.session_state.ingest_log[-25:]), language="log")
                else:
                    st.code("--:--:--  INFO  Awaiting trigger... (POST /ingest/run)", language="log")

        states = {k: ("done" if st.session_state.ingest_done else "", "") for k, _, _ in PIPE_STAGES}
        if st.session_state.ingest_done:
            r = st.session_state.ingest_stats["rows"]
            v = st.session_state.ingest_stats["vecs"]
            states["postgres"] = ("done", f"{r:,} rader".replace(",", " "))
            states["chunks"]   = ("done", f"{v:,} vec".replace(",", " "))
        with pipeline_box.container():
            render_pipeline(states)
        render_stats()
        render_log()

        if submitted:
            total_rows = (17476 if "2026-05-26" in source
                          else (17421 if "2026-04-12" in source else 17500))
            dim = 1024 if "e5-large" in model else 768

            st.session_state.ingest_log = []
            st.session_state.ingest_stats = {"rows": 0, "chunks": 0, "vecs": 0, "dim": dim}

            def log(level, msg):
                ts = time.strftime("%H:%M:%S")
                st.session_state.ingest_log.append(f"{ts}  {level:<4}  {msg}")
                render_log()

            log("INFO", f"POST /ingest/run · source={source.split()[0]} · text_column={text_col} · model={model.split()[0]}")

            stages = {k: ("", "") for k, _, _ in PIPE_STAGES}

            # 1 · JSON
            stages["json"] = ("active", "reading")
            with pipeline_box.container():
                render_pipeline(stages)
            progress_box.progress(0.10, text="Reading JSON...")
            time.sleep(0.4)
            log("OK", f"Read {total_rows:,} STIFTELSE records from JSON".replace(",", " "))
            stages["json"] = ("done", f"{total_rows:,} rec".replace(",", " "))

            # 2 · Polars
            stages["polars"] = ("active", "parsing")
            with pipeline_box.container():
                render_pipeline(stages)
            progress_box.progress(0.22, text="Polars · parsing 12 columns")
            time.sleep(0.4)
            log("INFO", "Polars DataFrame · 12 cols (id, namn, orgnr, andamal, ort, lankod, kommunkod, typ, ...)")
            stages["polars"] = ("done", f"{total_rows:,}×12".replace(",", " "))

            # 3 · Postgres COPY
            stages["postgres"] = ("active", "COPY")
            with pipeline_box.container():
                render_pipeline(stages)
            log("SQL", "DROP TABLE IF EXISTS stiftelser; CREATE TABLE stiftelser (...);")
            for i in range(1, 11):
                r = round(total_rows * i / 10)
                st.session_state.ingest_stats["rows"] = r
                stages["postgres"] = ("active", f"{r:,} rader".replace(",", " "))
                with pipeline_box.container():
                    render_pipeline(stages)
                render_stats()
                progress_box.progress(
                    0.22 + (i / 10) * 0.20,
                    text=f"COPY → stiftelser · {r:,} rader".replace(",", " "),
                )
                time.sleep(0.06)
            log("SQL", "COPY stiftelser FROM STDIN ·  done")
            log("OK", f"COPY complete · {total_rows:,} rows into 'stiftelser'".replace(",", " "))
            stages["postgres"] = ("done", f"{total_rows:,} rader".replace(",", " "))

            non_null = round(total_rows * 0.987)
            log("INFO", f"SELECT id, andamal FROM stiftelser WHERE andamal IS NOT NULL · {non_null:,} rows".replace(",", " "))

            # 4 · Chunk + embed
            total_chunks = round(non_null * 1.42 * (600 / max_chars))
            stages["chunk"] = ("active", "chunking")
            with pipeline_box.container():
                render_pipeline(stages)
            c = 0
            for i in range(1, 11):
                c = round(total_chunks * i / 10)
                st.session_state.ingest_stats["chunks"] = c
                stages["chunk"] = ("active", f"{c:,} chunks".replace(",", " "))
                with pipeline_box.container():
                    render_pipeline(stages)
                render_stats()
                progress_box.progress(
                    0.42 + (i / 10) * 0.20,
                    text=f"chunk_text() · {c:,} chunks".replace(",", " "),
                )
                time.sleep(0.05)
            log("OK", f"Chunked: {non_null:,} andamal → {c:,} chunks (max_chars={max_chars})".replace(",", " "))

            stages["chunk"] = ("active", f"embedding ({dim}d)")
            with pipeline_box.container():
                render_pipeline(stages)
            v = 0
            for i in range(1, 11):
                v = round(c * i / 10)
                st.session_state.ingest_stats["vecs"] = v
                stages["chunk"] = ("active", f"{v:,}/{c:,} vec".replace(",", " "))
                with pipeline_box.container():
                    render_pipeline(stages)
                render_stats()
                progress_box.progress(
                    0.62 + (i / 10) * 0.25,
                    text=f'"passage: " + chunk → e5 · {v:,} vec'.replace(",", " "),
                )
                time.sleep(0.05)
            log("INFO", f"e5 multilingual · {dim}d · {random.uniform(90, 150):.0f} vec/s")
            stages["chunk"] = ("done", f"{v:,} vec · {dim}d".replace(",", " "))

            # 5 · chunks upsert + HNSW
            stages["chunks"] = ("active", "upserting")
            with pipeline_box.container():
                render_pipeline(stages)
            log("SQL", "INSERT INTO chunks (source_id, source_table, source_column, chunk_index, "
                        "chunk_text, embedding, model_name) ON CONFLICT DO UPDATE")
            progress_box.progress(0.92, text="upsert chunks")
            time.sleep(0.4)
            if hnsw:
                stages["chunks"] = ("active", "build HNSW")
                with pipeline_box.container():
                    render_pipeline(stages)
                progress_box.progress(0.96, text="CREATE INDEX ... USING hnsw")
                time.sleep(0.6)
                log("SQL", "CREATE INDEX chunks_emb_hnsw ON chunks USING hnsw "
                            "(embedding vector_cosine_ops) WITH (m=16, ef_construction=64);")
            stages["chunks"] = ("done", f"{v:,} vec indexed".replace(",", " "))
            with pipeline_box.container():
                render_pipeline(stages)

            if verify:
                time.sleep(0.35)
                log("OK", 'Verify · sample query "hemlöshet" → top-5 chunks · mean cosine 0.81 ✓')
            if enrich:
                log("WARN", "v2 · table-enricher endast simulation (decode, scrape, klassificera)")

            progress_box.progress(1.0, text="COMPLETE")
            log("OK", f"Pipeline complete · {total_rows:,} rows · {v:,} chunks · {dim}d".replace(",", " "))

            st.session_state.ingest_done = True
            render_stats()
            st.success(f"Phase 1 klar · {v:,} chunks · {dim}d · Postgres + pgvector + HNSW".replace(",", " "))


# -----------------------------------------------------------------------------
# TAB 2 · MATCH & DRAFT
# -----------------------------------------------------------------------------
with tab2:
    left, right = st.columns([1, 2.4], gap="large")

    with left:
        st.markdown("##### Query-konfiguration")

        prog_key = st.selectbox(
            "Stadsmissions-verksamhet",
            options=list(PROGRAMMES.keys()),
            format_func=lambda k: PROGRAMMES[k]["label"],
            help="Verksamhetsprofil = query för cosine-sökning",
        )
        prog_text = st.text_area(
            'Profil-text  (embeddas som "query: " + text)',
            value=PROGRAMMES[prog_key]["text"],
            height=160,
        )
        top_k = st.slider("top-K chunks (ANN)", 5, 60, 25, 5, help="cosine över HNSW · grov chunk-grain")
        agg = st.radio("Aggregering chunks → stiftelse", ["mean", "max", "sum"], horizontal=True, index=0)
        top_n = st.slider("Top-N efter group-by", 3, 30, 10, 1)

        st.markdown("**LLM-steg**")
        rerank = st.toggle("LLM rerank över hela ANDAMAL", value=True)
        geo_sthlm = st.toggle("Geo-filter · Stockholms län", value=True)
        typ6 = st.toggle("TYP = 6 (allmännyttig)", value=True)
        fitgap = st.toggle("Fit & gap-analys (1 LLM-call/stiftelse)", value=True)
        do_draft = st.toggle("Generera ansökningsutkast (~200 ord)", value=True)
        final_k = st.slider("Slutgiltigt antal förslag", 1, 6, 3, 1)

        run = st.button("▶ Matcha & generera utkast", type="primary", use_container_width=True)
        st.caption("stateless · idempotent · v1")

    with right:
        st.markdown("##### Pipeline-utfall")
        out_status = st.empty()
        summary_box = st.empty()
        results_box = st.empty()

        if run:
            t0 = time.time()
            pool = list(STIFTELSER)
            if geo_sthlm:
                pool = [s for s in pool if s["lankod"] == 1]
            if typ6:
                pool = [s for s in pool if s["typ"] == 6]

            with out_status.container():
                with st.status("Embedding query...", expanded=False) as status:
                    status.update(label='embed("query: ' + PROGRAMMES[prog_key]["label"].lower() + ' ...")')
                    time.sleep(0.30)
                    status.update(label="ANN search · top-K = " + str(top_k) + " chunks · cosine over HNSW")
                    time.sleep(0.35)
                    status.update(label="GROUP BY stiftelse_id · aggregate (" + agg + ")")
                    time.sleep(0.25)
                    scored = sorted(
                        [{**s, "_s": score_for(s, prog_key, agg, rerank)} for s in pool],
                        key=lambda x: -x["_s"],
                    )
                    candidates = scored[: min(top_n, len(scored))]
                    if rerank:
                        status.update(label="LLM rerank · rubric över hela ANDAMAL · k=" + str(len(candidates)))
                        time.sleep(0.55)
                    final_set = candidates[:final_k]
                    status.update(label="Fit+gap × " + str(len(final_set)) + " · composing drafts", state="complete")

            elapsed_ms = int((time.time() - t0) * 1000)

            with summary_box.container():
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Programme", PROGRAMMES[prog_key]["label"])
                c2.metric("Chunks hämtade", top_k)
                c3.metric("Kandidater (group-by)", len(candidates))
                c4.metric("Visas (slutligt)", len(final_set), str(elapsed_ms) + " ms")

            # render foundation cards — ONE st.markdown per card to avoid markdown-parser
            # interleaving issues with multi-line HTML / mixed components
            with results_box.container():
                for i, f in enumerate(final_set):
                    rerank_score = f["_s"] + 0.02 - i * 0.01
                    orig_rank = next(
                        (j for j, x in enumerate(candidates) if x["id"] == f["id"]), i
                    ) + 1
                    rerank_shift = (
                        " &nbsp;·&nbsp; ← #" + str(orig_rank)
                        if (rerank and orig_rank != i + 1)
                        else ""
                    )

                    andamal_html = highlight_andamal(f["andamal"], f["highlight"])

                    if fitgap:
                        fg = compose_fit_gap(prog_key, f)
                        aligned_lis = "".join("<li>" + x + "</li>" for x in fg["aligned"])
                        gap_lis     = "".join("<li>" + x + "</li>" for x in fg["gaps"])
                        reframe_lis = "".join("<li>" + x + "</li>" for x in fg["reframe"])
                        fitgap_block = (
                            '<div class="sec-h">Fit · Gap · Reframe (1 LLM-call)</div>'
                            '<div class="fitgap">'
                            '<div class="aligned"><div class="hh">✓ Aligned</div><ul>'
                            + aligned_lis + '</ul></div>'
                            '<div class="gap"><div class="hh">⚠ Gap</div><ul>'
                            + gap_lis + '</ul></div>'
                            '<div class="reframe"><div class="hh">↻ Reframe</div><ul>'
                            + reframe_lis + '</ul></div>'
                            '</div>'
                        )
                    else:
                        fitgap_block = ""

                    if do_draft:
                        draft_inner = compose_draft(prog_key, f)
                        wc = len(re.sub(r"<[^>]+>", "", draft_inner).split())
                        draft_block = (
                            '<div class="sec-h">Ansökningsutkast · ~200 ord</div>'
                            '<div class="draft">' + draft_inner + '</div>'
                            '<div class="draft-meta">' + str(wc)
                            + ' ord · generated by claude-sonnet-4-6 · stateless</div>'
                        )
                    else:
                        draft_block = ""

                    apptype = suggest_project_type(prog_key, f)
                    prog_label = PROGRAMMES[prog_key]["label"]

                    # Build the whole card as ONE flat string — no leading whitespace, no embedded
                    # newlines, so Streamlit's markdown parser can't mistake parts of it for code
                    # blocks or split it across paragraphs.
                    card_html = (
                        '<div class="fcard">'
                          '<div style="display:flex;align-items:flex-start;gap:14px">'
                            '<div class="rank-badge">#' + str(i + 1) + '</div>'
                            '<div style="flex:1;min-width:0">'
                              '<h4>' + f["namn"] + '</h4>'
                              '<div style="margin-bottom:6px">'
                                '<span class="badge">org.nr ' + f["orgnr"] + '</span>'
                                '<span class="badge">ID ' + str(f["id"]) + '</span>'
                                '<span class="badge lan">'
                                + LAN.get(f["lankod"], "?") + ' · '
                                + KOMMUN.get(f["kommunkod"], f["ort"]) + '</span>'
                                '<span class="badge">TYP ' + str(f["typ"]) + ' · '
                                + TYP.get(f["typ"], "?") + '</span>'
                                '<span class="badge">storlek: ' + f["size"] + '</span>'
                              '</div>'
                              '<div class="scorebox">'
                                'agg <b>' + ('%.3f' % f["_s"]) + '</b>'
                                ' &nbsp;·&nbsp; rerank <b>' + ('%.3f' % rerank_score) + '</b>'
                                + rerank_shift +
                              '</div>'
                            '</div>'
                          '</div>'
                          '<div class="sec-h">ANDAMAL · matchade fragment</div>'
                          '<div class="andamal">' + andamal_html + '</div>'
                          '<div class="notes">' + f["notes"] + '</div>'
                          + fitgap_block +
                          '<div class="sec-h">Föreslagen ansökningstyp</div>'
                          '<div class="apptype"><b>' + prog_label + '</b> &nbsp;→&nbsp; '
                          + apptype + '</div>'
                          + draft_block +
                          '<div class="sec-h">Ansökningsmetod</div>'
                          '<div class="v2">app-method block · email / form / pdf · '
                          'contact · deadline · (kommer i v2 från table-enricher)</div>'
                        '</div>'
                    )

                    st.markdown(card_html, unsafe_allow_html=True)
        else:
            results_box.info(
                "← välj verksamhet och kör pipelinen för att se top-N stiftelser, fit/gap och utkast"
            )


# =============================================================================
# FOOTER
# =============================================================================
st.divider()
fcols = st.columns(6)
fcols[0].caption("DB · postgres://stadsmission")
fcols[1].caption("pgvector 0.7.4 · HNSW")
fcols[2].caption("FastHTML v0.6.x")
fcols[3].caption("LLM · claude-sonnet-4-6")
fcols[4].caption("v0.4.1 · stateless")
fcols[5].caption("apassov@stadsmissionen.se")
