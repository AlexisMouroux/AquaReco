"""
Enrichissement OpenStreetMap - AquaReco
Récupère les équipements de proximité des sites de baignade via l'API Overpass
(gratuite, sans clé, requêtes QL POST).

Équipements recherchés et rayons réglementaires :
  parking       amenity=parking                    500 m
  sanitaires    amenity=toilets                    200 m
  pmr           wheelchair=yes/designated          200 m
  douche        amenity=shower                     200 m
  poste_secours emergency=lifeguard*               300 m

Stratégie d'appel API :
  • Démo (n=5) : une requête `around:` par site → immédiat, sans cache
  • Run complet  : une requête bbox par cellule 0.5° (≈60 km)
                   → ~60-80 requêtes pour couvrir France + DOM-TOM
                   → résultat mis en cache dans outputs/osm_amenities_cache.parquet
"""

import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

# ── Configuration ──────────────────────────────────────────────────────────────

OVERPASS_ENDPOINTS = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass-api.de/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]

RADII_M = {
    "parking":       500,
    "sanitaires":    200,
    "pmr":           200,
    "douche":        200,
    "poste_secours": 300,
}
MAX_RADIUS_M = max(RADII_M.values())    # 500 m

COORD_ROUND   = 0.5       # cellules 0.5° ≈ 55 km × 38 km en France métropolitaine
BBOX_BUFFER   = 0.006     # marge autour de la bbox d'une cellule (~660 m ≈ MAX_RADIUS_M)
REQUEST_DELAY = 1.0       # secondes entre deux requêtes (fair use Overpass)

OUTPUT_DIR  = Path(__file__).parent / "outputs"
CACHE_FILE  = OUTPUT_DIR / "osm_amenities_cache.parquet"
EQUIP_FILE  = OUTPUT_DIR / "osm_equipements.csv"


# ── Requêtes Overpass QL ───────────────────────────────────────────────────────

# Pour le run complet : bounding box couvrant une cellule de sites
_QL_BBOX = """
[out:json][timeout:90];
(
  node["amenity"~"^(parking|toilets|shower)$"]({s},{w},{n},{e});
  way["amenity"~"^(parking|toilets|shower)$"]({s},{w},{n},{e});
  node["wheelchair"~"^(yes|designated)$"]({s},{w},{n},{e});
  node["emergency"~"^(lifeguard|lifeguard_base|lifeguard_tower)$"]({s},{w},{n},{e});
  way["emergency"~"^(lifeguard|lifeguard_base|lifeguard_tower)$"]({s},{w},{n},{e});
);
out center;
"""

# Pour la démo : around: centré sur le site (plus précis mais 1 requête / site)
_QL_AROUND = """
[out:json][timeout:30];
(
  node["amenity"~"^(parking|toilets|shower)$"](around:{r},{lat},{lon});
  way["amenity"~"^(parking|toilets|shower)$"](around:{r},{lat},{lon});
  node["wheelchair"~"^(yes|designated)$"](around:{r},{lat},{lon});
  node["emergency"~"^(lifeguard|lifeguard_base|lifeguard_tower)$"](around:{r},{lat},{lon});
  way["emergency"~"^(lifeguard|lifeguard_base|lifeguard_tower)$"](around:{r},{lat},{lon});
);
out center;
"""


# ── Utilitaires ────────────────────────────────────────────────────────────────

def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance haversine en mètres entre deux points WGS84."""
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2.0 * R * math.asin(math.sqrt(min(1.0, a)))


def _fetch_overpass(query: str, retries: int = 3) -> dict | None:
    """
    Envoie une requête Overpass QL (POST).
    Essaie plusieurs endpoints, applique un backoff exponentiel sur les erreurs 429.
    Retourne None si tous les essais échouent.
    """
    for endpoint in OVERPASS_ENDPOINTS:
        for attempt in range(retries):
            try:
                resp = requests.post(
                    endpoint, data={"data": query}, timeout=120
                )
                if resp.status_code == 429:
                    wait = 30 * (attempt + 1)
                    logger.warning("429 Too Many Requests — attente %d s...", wait)
                    time.sleep(wait)
                    continue
                if resp.status_code == 406:
                    # IP bloquee sur cet endpoint, inutile de reessayer
                    logger.debug("406 sur %s, bascule vers endpoint suivant.", endpoint)
                    break
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as e:
                logger.warning("Erreur %s (tentative %d) : %s", endpoint, attempt + 1, e)
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
        else:
            logger.warning("Endpoint %s epuise, bascule.", endpoint)
    return None


def _parse_elements(data: dict) -> pd.DataFrame:
    """Convertit la réponse JSON Overpass en DataFrame plat avec lat, lon et tags clés."""
    records = []
    for elem in data.get("elements", []):
        # Les ways ont leur centroïde dans "center" avec `out center;`
        lat = elem.get("lat") or (elem.get("center") or {}).get("lat")
        lon = elem.get("lon") or (elem.get("center") or {}).get("lon")
        if lat is None or lon is None:
            continue
        tags = elem.get("tags", {})
        records.append({
            "osm_type":   elem["type"],
            "osm_id":     elem["id"],
            "lat":        float(lat),
            "lon":        float(lon),
            "amenity":    tags.get("amenity", ""),
            "emergency":  tags.get("emergency", ""),
            "wheelchair": tags.get("wheelchair", ""),
        })
    _EMPTY_COLS = ["osm_type", "osm_id", "lat", "lon", "amenity", "emergency", "wheelchair"]
    return pd.DataFrame(records) if records else pd.DataFrame(columns=_EMPTY_COLS)


def _amenity_mask(equip: str, df: pd.DataFrame) -> pd.Series:
    """Masque booléen des éléments correspondant à la catégorie d'équipement."""
    if equip == "parking":
        return df["amenity"] == "parking"
    if equip == "sanitaires":
        return df["amenity"] == "toilets"
    if equip == "pmr":
        return df["wheelchair"].isin(["yes", "designated"])
    if equip == "douche":
        return df["amenity"] == "shower"
    if equip == "poste_secours":
        return df["emergency"].isin(["lifeguard", "lifeguard_base", "lifeguard_tower"])
    return pd.Series(False, index=df.index)


def _check_site(lat: float, lon: float, amenities: pd.DataFrame) -> dict:
    """
    Retourne {équipement: bool} pour un site, en testant la présence de chaque
    catégorie d'équipement dans son rayon réglementaire (haversine).
    """
    result = {}
    for equip, radius in RADII_M.items():
        if amenities.empty:
            result[equip] = False
            continue
        mask = _amenity_mask(equip, amenities)
        cands = amenities[mask]
        if cands.empty:
            result[equip] = False
        else:
            dists = cands.apply(
                lambda r: _haversine(lat, lon, r["lat"], r["lon"]), axis=1
            )
            result[equip] = bool((dists <= radius).any())
    return result


# ── Clustering géographique (run complet) ─────────────────────────────────────

def _cluster_sites(df_sites: pd.DataFrame) -> pd.DataFrame:
    """
    Affecte chaque site unique à une cellule (lat_r, lon_r) arrondie à COORD_ROUND.
    Retourne un DataFrame avec code_site, lat_r, lon_r.
    """
    df = (df_sites[["code_site", "latitude", "longitude"]]
          .drop_duplicates("code_site")
          .dropna(subset=["latitude", "longitude"])
          .copy())
    p = COORD_ROUND
    df["lat_r"] = (df["latitude"]  / p).round() * p
    df["lon_r"] = (df["longitude"] / p).round() * p
    return df[["code_site", "lat_r", "lon_r", "latitude", "longitude"]]


# ── Cache Overpass (run complet) ───────────────────────────────────────────────

def build_amenities_cache(df_sites: pd.DataFrame,
                           use_cache: bool = True) -> pd.DataFrame:
    """
    Récupère (ou charge depuis le cache Parquet) les équipements OSM pour toutes
    les cellules géographiques couvrant les sites fournis.

    Stratégie : une requête bbox par cellule 0.5° → ~60-80 requêtes pour la France.
    Le cache est sauvegardé dans outputs/osm_amenities_cache.parquet.

    Retourne un DataFrame avec colonnes :
        lat_r, lon_r, osm_type, osm_id, lat, lon, amenity, emergency, wheelchair
    """
    OUTPUT_DIR.mkdir(exist_ok=True)

    if use_cache and CACHE_FILE.exists():
        logger.info("Cache OSM charge : %s", CACHE_FILE)
        return pd.read_parquet(CACHE_FILE)

    clusters_df = _cluster_sites(df_sites)
    cells = (
        clusters_df
        .groupby(["lat_r", "lon_r"])
        .agg(
            s=("latitude",  "min"),
            n=("latitude",  "max"),
            w=("longitude", "min"),
            e=("longitude", "max"),
            n_sites=("code_site", "count"),
        )
        .reset_index()
    )
    n_cells = len(cells)
    logger.info("%d cellules OSM a interroger.", n_cells)

    all_parts = []
    for i, (_, row) in enumerate(cells.iterrows(), 1):
        # Bbox = extent des sites dans la cellule + marge de sécurité
        s = row["s"] - BBOX_BUFFER
        n = row["n"] + BBOX_BUFFER
        w = row["w"] - BBOX_BUFFER
        e = row["e"] + BBOX_BUFFER

        logger.info("  [%d/%d] lat_r=%.1f lon_r=%.1f  (%d sites)  bbox=[%.3f,%.3f,%.3f,%.3f]",
                    i, n_cells, row.lat_r, row.lon_r, int(row.n_sites), s, w, n, e)

        query = _QL_BBOX.format(s=s, w=w, n=n, e=e)
        data  = _fetch_overpass(query)

        if data is None:
            logger.warning("  -> Echec, cellule ignoree.")
        else:
            df_elem = _parse_elements(data)
            if not df_elem.empty:
                df_elem["lat_r"] = row.lat_r
                df_elem["lon_r"] = row.lon_r
                all_parts.append(df_elem)
            logger.info("  -> %d elements recuperes.", len(df_elem) if data else 0)

        time.sleep(REQUEST_DELAY)

    if not all_parts:
        logger.warning("Aucun element OSM recupere — verifier la connectivite.")
        return pd.DataFrame(
            columns=["lat_r", "lon_r", "osm_type", "osm_id",
                     "lat", "lon", "amenity", "emergency", "wheelchair"]
        )

    cache = pd.concat(all_parts, ignore_index=True)
    cache.to_parquet(CACHE_FILE, index=False)
    logger.info("Cache OSM sauvegarde : %s  (%d elements)", CACHE_FILE, len(cache))
    return cache


# ── Pipeline complet (run complet) ────────────────────────────────────────────

def fetch_osm_equipements(df_sites: pd.DataFrame,
                           use_cache: bool = True) -> pd.DataFrame:
    """
    Pipeline OSM complet.

    1. Construit le cache de proximité (batch par cellule 0.5°)
    2. Pour chaque site unique, vérifie la présence de chaque équipement
       par haversine dans son rayon réglementaire
    3. Exporte outputs/osm_equipements.csv et retourne le DataFrame

    Paramètres
    ----------
    df_sites  : DataFrame consolidé site × saison (issu de etl.py)
    use_cache : True = réutilise le cache Parquet si présent

    Colonnes retournées : code_site, parking, sanitaires, pmr, douche, poste_secours
    """
    sites_uniq = (
        df_sites[["code_site", "latitude", "longitude"]]
        .drop_duplicates("code_site")
        .dropna(subset=["latitude", "longitude"])
        .reset_index(drop=True)
    )
    logger.info("Verification equipements OSM pour %d sites uniques.", len(sites_uniq))

    cache = build_amenities_cache(df_sites, use_cache=use_cache)

    clusters_df = _cluster_sites(df_sites)
    sites_uniq  = sites_uniq.merge(
        clusters_df[["code_site", "lat_r", "lon_r"]], on="code_site", how="left"
    )

    # Index du cache par cellule pour éviter un groupby répété à chaque site
    if not cache.empty:
        cache_by_cell = {
            key: grp.reset_index(drop=True)
            for key, grp in cache.groupby(["lat_r", "lon_r"])
        }
    else:
        cache_by_cell = {}

    results = []
    for _, row in sites_uniq.iterrows():
        cell_am = cache_by_cell.get((row["lat_r"], row["lon_r"]), pd.DataFrame())
        equip   = _check_site(row["latitude"], row["longitude"], cell_am)
        equip["code_site"] = row["code_site"]
        results.append(equip)

    df_equip = pd.DataFrame(results)[
        ["code_site", "parking", "sanitaires", "pmr", "douche", "poste_secours"]
    ]

    OUTPUT_DIR.mkdir(exist_ok=True)
    df_equip.to_csv(EQUIP_FILE, index=False, sep=";", encoding="utf-8-sig")
    logger.info("Equipements OSM exportes : %s  (%d sites)", EQUIP_FILE, len(df_equip))

    print("\n--- Statistiques equipements (% sites equipes) ---")
    for col in ["parking", "sanitaires", "pmr", "douche", "poste_secours"]:
        pct = df_equip[col].mean() * 100
        bar = "#" * int(pct / 2)
        print(f"  {col:<16} {bar:<50} {pct:5.1f}%")

    return df_equip


# ── Démo 5 sites (requêtes individuelles around:) ─────────────────────────────

def run_demo(df_sites: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    Teste l'enrichissement OSM sur n sites représentatifs.

    Utilise des requêtes Overpass `around:MAX_RADIUS` individuelles pour
    chaque site (pas de cache - clairement lisible pour la démo).
    Choisit des sites de types variés : côtier, lac, rivière, DOM-TOM.
    """
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    def _pick(keyword: str, exclude_codes: set) -> pd.Series | None:
        """Retourne le site le plus prélevé dont le type_eau contient keyword."""
        import unicodedata as _ud
        def _norm(s):
            s = _ud.normalize("NFD", str(s)).encode("ascii", "ignore").decode("ascii")
            return s.lower()
        mask = (
            df_sites["type_eau"].map(lambda x: keyword in _norm(x) if pd.notna(x) else False)
            & ~df_sites["code_site"].isin(exclude_codes)
        )
        cands = df_sites[mask].drop_duplicates("code_site")
        cands = cands.dropna(subset=["latitude", "longitude"])
        if cands.empty:
            return None
        return cands.iloc[0]

    # Sélectionne n sites avec des types d'eau variés
    seen: set = set()
    categories = [
        ("cotiere",  "Cote"),
        ("lac",      "Lac"),
        ("riviere",  "Riviere"),
        ("mer",      "Mer"),
        ("transit",  "Transition"),
    ]
    sample_rows = []
    for kw, label in categories:
        if len(sample_rows) >= n:
            break
        row = _pick(kw, seen)
        if row is not None:
            row = row.copy()
            row["_label"] = label
            sample_rows.append(row)
            seen.add(row["code_site"])

    # Complète avec les premiers sites disponibles si besoin
    for _, row in df_sites.drop_duplicates("code_site").iterrows():
        if len(sample_rows) >= n:
            break
        if row["code_site"] not in seen and pd.notna(row.get("latitude")):
            row = row.copy()
            row["_label"] = str(row.get("type_eau", "?"))
            sample_rows.append(row)
            seen.add(row["code_site"])

    print("\n" + "=" * 72)
    print(f"  DEMO osm.py — {len(sample_rows)} sites (requetes around:{MAX_RADIUS_M}m)")
    print("=" * 72)
    print(f"  {'#':<2}  {'Nom du site':<32} {'Type':<14} {'Park':>5} "
          f"{'WC':>5} {'PMR':>5} {'Dch':>5} {'Sec':>5}")
    print("  " + "-" * 72)

    results = []
    for idx, row in enumerate(sample_rows, 1):
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        nom = str(row.get("nom_site", "?"))
        label = str(row.get("_label", "?"))

        query = _QL_AROUND.format(r=MAX_RADIUS_M, lat=lat, lon=lon)
        data  = _fetch_overpass(query)

        if data is not None:
            amenities = _parse_elements(data)
            n_elems   = len(amenities)
        else:
            amenities = pd.DataFrame()
            n_elems   = 0

        equip = _check_site(lat, lon, amenities)

        def _b(v): return "  Y" if v else "  ."

        print(f"  {idx:<2}  {nom[:31]:<32} {label[:13]:<14}"
              f"{_b(equip['parking'])}{_b(equip['sanitaires'])}"
              f"{_b(equip['pmr'])}{_b(equip['douche'])}"
              f"{_b(equip['poste_secours'])}"
              f"   [{n_elems} elem. OSM]")

        # Détail des équipements trouvés (pour les 2 premiers sites)
        if idx <= 2 and not amenities.empty:
            for eq, radius in RADII_M.items():
                mask  = _amenity_mask(eq, amenities)
                cands = amenities[mask]
                if cands.empty:
                    continue
                dists = cands.apply(
                    lambda r: _haversine(lat, lon, r["lat"], r["lon"]), axis=1
                )
                nearby = cands[dists <= radius]
                if not nearby.empty:
                    closest = dists[dists <= radius].min()
                    print(f"       {eq:<14} : {len(nearby)} objet(s), plus proche a {closest:.0f} m")

        equip["code_site"] = row["code_site"]
        equip["nom_site"]  = nom
        results.append(equip)
        time.sleep(REQUEST_DELAY)

    print(f"\n  Legende : Y = present   . = absent")
    print(f"  Rayons  : parking {RADII_M['parking']}m | WC/douche/PMR "
          f"{RADII_M['sanitaires']}m | secours {RADII_M['poste_secours']}m")
    print()

    df_out = pd.DataFrame(results)[
        ["code_site", "nom_site", "parking", "sanitaires", "pmr", "douche", "poste_secours"]
    ]
    return df_out


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from etl import build_consolidated
    df_site_year, _ = build_consolidated()
    run_demo(df_site_year, n=5)
