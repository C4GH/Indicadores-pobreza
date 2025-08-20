import os 
import re
import numpy as np
import pandas as pd

# Configuración de rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "out")
os.makedirs(OUT_DIR, exist_ok=True)

# Archivos de entrada 
PATH_HOGARES    = os.path.join(DATA_DIR, "hogares.csv")
PATH_POBLACION  = os.path.join(DATA_DIR, "poblacion.csv")
PATH_TRABAJOS   = os.path.join(DATA_DIR, "trabajos.csv")
PATH_VIVIENDAS  = os.path.join(DATA_DIR, "viviendas.csv")
PATH_SINCO      = os.path.join(DATA_DIR, "sinco_interes_cultural.csv")

# Utilidades
def read_csv_safe(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, dtype=str, encoding="utf-8", low_memory=False)
    except Exception:
        return pd.read_csv(path, dtype=str, encoding="latin-1", low_memory=False)
    
def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def ambito_from_folioviv(fv):
    s = str(fv)
    if len(s) != 10 or not s.isdigit():
        return None  
    return "rural" if s[2] == "6" else "urbano"

def wmean(indic, w):
    w = pd.to_numeric(w, errors="coerce")
    x = pd.to_numeric(indic, errors ="coerce")
    sw = w.sum()
    return float((x * w).sum() / sw * 100) if sw > 0 else np.nan

# Carga
hogares     = norm_cols(read_csv_safe(PATH_HOGARES))
poblacion   = norm_cols(read_csv_safe(PATH_POBLACION))
trabajos    = norm_cols(read_csv_safe(PATH_TRABAJOS))
viviendas   = norm_cols(read_csv_safe(PATH_VIVIENDAS))
sinco       = norm_cols(read_csv_safe(PATH_SINCO))

# Validaciones mínimas
for name, df in [("hogares", hogares), ("poblacion", poblacion), ("trabajos", trabajos)]:
    for k in ["folioviv", "foliohog"]:
        if k not in df.columns:
            raise KeyError(f"Falta la llave '{k}' en {name}.")
if "numren" not in poblacion.columns or "numren" not in trabajos.columns:
    raise KeyError ("Falta 'numren' en POBLACION/TRABAJOS. ")
if "factor" not in poblacion.columns and "factor" not in trabajos.columns:
    raise KeyError("Se espera 'factor' (peso) en POBLACION o TRABAJOS (ENIGH 2024 NS).")

# cast pesos
for df in (hogares, poblacion, trabajos):
    if "factor" in df.columns:
        df["factor"] = pd.to_numeric(df["factor"], errors="coerce")

# Ocupaciones culturales(SINCO)
if "sinco" not in trabajos.columns:
    cand = [c for c in trabajos.columns if re.search(r"\bsinco\b|ocup", c)]
    if not cand:
        raise KeyError("No se encontró columna 'sinco' ni alternativa en TRABAJOS")
    trabajos.rename(columns={cand[0]: "sinco"}, inplace=True)

if not {"clave", "cultural"}.issubset(sinco.columns):
    raise KeyError("El Archivo sinco_interes_cultural.csv debe de tener columnas 'clave' y 'cultural'.")

trabajos["sinco"]   = trabajos["sinco"].astype(str).str.strip()
sinco["clave"]      = sinco["clave"].astype(str).str.strip()
sinco["cultural"]   = sinco["cultural"].astype(str).str.strip()

trab_cultura = trabajos.merge(
    sinco[["clave", "cultural"]],
    left_on="sinco", right_on="clave", how="left"
)
trab_cultura["_es_cultura"] = trab_cultura["cultural"] == "1"

# Jefes de hogar
CODIGO_JEFE = {"101"}
poblacion["_es_jefe"] = poblacion["parentesco"].astype(str).str.zfill(3).isin(CODIGO_JEFE)

# Unión: Jefes + Cultura
cj = trab_cultura.merge(
    poblacion[["folioviv", "foliohog", "numren", "_es_jefe", "entidad"] + (["factor"] if "factor" in poblacion.columns else [])],
    on = ["folioviv", "foliohog", "numren"], how="left"
)

# Coalesce de pesos
fac_cols = [c for c in cj.columns if re.fullmatch(r"factor(_x|_y)?", c)]
if not fac_cols:
    raise KeyError("Después del merge no se encontró ninguna columna de factor en 'cj'.")
cj["factor_w"] = pd.to_numeric(cj[fac_cols].bfill(axis=1).iloc[:, 0], errors="coerce")

# Filtrar jeffes + cultura
cj = cj[(cj["_es_cultura"]) & (cj["_es_jefe"])].copy()
if cj.empty:
    raise ValueError("No se encontraron jefes de hogar con ocupaciones culturales")

# Proxies de inseguridad alimentaria (desde HOGARES acc_alim*)
acc_cols = [c for c in hogares.columns if c.startswith("acc_alim")]
cj = cj.merge(hogares[["folioviv", "foliohog"] + acc_cols], on=["folioviv", "foliohog"], how="left")

for v in ["acc_alim13", "acc_alim14", "acc_alim15", "acc_alim7", "acc_alim8"]:
    if v not in cj.columns:
        cj[v] = np.nan

cj["_insegalim_severa_proxy"] = (
    (cj["acc_alim13"] == "1") | (cj["acc_alim14"] == "1") | (cj["acc_alim15"] == "1")
).astype(int)

cj["_insegalim_modsev_proxy"] = (
    (cj["_insegalim_severa_proxy"] == 1) | (cj["acc_alim7"] == "1") | (cj["acc_alim8"] == "1")
).astype(int)

# Ámbito urbano/rural desde folioviv
cj["ambito"] = cj["folioviv"].astype(str).map(ambito_from_folioviv)

# Tamaño del hogar
tam_hogar = poblacion.groupby(["folioviv", "foliohog"])["numren"].count().reset_index()
tam_hogar.rename(columns={"numren": "tam_hogar"}, inplace=True)

# Unir al cj (solo jefes culturales)
cj = cj.merge(tam_hogar, on=["folioviv", "foliohog"], how="left")

# Tabulados ponderados (usar 'factor_w')
frames = []

# Nacional 
tot_w = cj["factor_w"].sum()
df_nat = pd.DataFrame({
    "nivel": ["Nacional"],
    "grupo": ["Total"],
    "personas_ocup_cultura_jefes": [tot_w],
    "tam_hogar_promedio": [np.average(cj["tam_hogar"], weights=cj["factor_w"])],
    "insegalim_mod_sev_%": [wmean(cj["_insegalim_modsev_proxy"], cj["factor_w"])],
    "insegalim_sev_%": [wmean(cj["_insegalim_severa_proxy"], cj["factor_w"])],
})
frames.append(df_nat)

# Por entidad
if "entidad" in cj.columns:
    g = cj.groupby("entidad", dropna=False)
    df_ent = pd.DataFrame({
        "nivel": "Por entidad",
        "grupo": g.apply(lambda d: d.name),
        "personas_ocup_cultura_jefes": g["factor_w"].sum().values,
        "tam_hogar_promedio": g.apply(lambda d: np.average(d["tam_hogar"], weights=d["factor_w"])),
        "insegalim_mod_sev_%": g.apply(lambda d: wmean(d["_insegalim_modsev_proxy"], d["factor_w"])).values,
        "insegalim_sev_%": g.apply(lambda d: wmean(d["_insegalim_severa_proxy"], d["factor_w"])).values,
    })
    frames.append(df_ent)

# Por ámbito
g = cj.groupby("ambito", dropna=False)
df_amb = pd.DataFrame({
    "nivel": "Por ámbito",
    "grupo": g.apply(lambda d: d.name),
    "personas_ocup_cultura_jefes": g["factor_w"].sum().values,
    "tam_hogar_promedio": g.apply(lambda d: np.average(d["tam_hogar"], weights=d["factor_w"])).values,
    "insegalim_mod_sev_%": g.apply(lambda d: wmean(d["_insegalim_modsev_proxy"], d["factor_w"])).values,
    "insegalim_sev_%":    g.apply(lambda d: wmean(d["_insegalim_severa_proxy"], d["factor_w"])).values,
})
frames.append(df_amb)

# Por entidad y ámbito
if "entidad" in cj.columns:
    g = cj.groupby(["entidad", "ambito"], dropna=False)
    df_ea = pd.DataFrame({
        "nivel": "Por entidad y ámbito",
        "grupo": g.apply(lambda d: f"{d.name[0]} - {d.name[1]}"),
        "personas_ocup_cultura_jefes": g["factor_w"].sum().values,
        "insegalim_mod_sev_%": g.apply(lambda d: wmean(d["_insegalim_modsev_proxy"], d["factor_w"])).values,
        "insegalim_sev_%":    g.apply(lambda d: wmean(d["_insegalim_severa_proxy"], d["factor_w"])).values,
    })
    frames.append(df_ea)

resumen = pd.concat(frames, ignore_index=True)

# Salidas
resumen_path = os.path.join(OUT_DIR, "resumen_cultura_jefes.csv")
micro_path = os.path.join(OUT_DIR, "micro_cultura_jefes.csv")

resumen.to_csv(resumen_path, index=False, encoding="utf-8")

cols_micro = ["folioviv", "foliohog", "numren", "entidad", "ambito", "sinco", "factor_w",
              "_insegalim_modsev_proxy", "_insegalim_severa_proxy"]
cols_micro = [c for c in cols_micro if c in cj.columns]
cj[cols_micro].to_csv(micro_path, index=False, encoding="utf-8")

print("Archivos generados: ")
print(" -", resumen_path)
print(" -", micro_path)