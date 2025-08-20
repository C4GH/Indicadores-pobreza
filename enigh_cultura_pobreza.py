import os
import re
import numpy as np
import pandas as pd

# Configuración de rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR  = os.path.join(BASE_DIR, "out")
os.makedirs(OUT_DIR, exist_ok=True)

ANIO_ENCUESTA = 2024             # tu año de ENIGH (si lo necesitas para rotular)
ANIO_LINEAS   = 2025             # <-- líneas de pobreza FIJAS a 2025 (enero)

# Archivos de entrada
PATH_HOGARES   = os.path.join(DATA_DIR, "hogares.csv")
PATH_POBLACION = os.path.join(DATA_DIR, "poblacion.csv")
PATH_TRABAJOS  = os.path.join(DATA_DIR, "trabajos.csv")
PATH_VIVIENDAS = os.path.join(DATA_DIR, "viviendas.csv")
PATH_SINCO     = os.path.join(DATA_DIR, "sinco_interes_cultural.csv")
PATH_CONC      = os.path.join(DATA_DIR, "concentradohogar.csv")        # ingresos del hogar
PATH_LINEAS    = os.path.join(DATA_DIR, "lineas_pobreza_coneval.csv")  # anio,mes,fecha,ambito,tipo,monto


# Utilidades
def read_csv_safe(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, dtype=str, encoding="utf-8", low_memory=False)
    except Exception:
        return pd.read_csv(path, dtype=str, encoding="latin-1", low_memory=False)

def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def ambito_from_folioviv(fv: str):
    s = str(fv)
    if len(s) != 10 or not s.isdigit():
        return None
    return "rural" if s[2] == "6" else "urbano"

def wmean(indic, w):
    w = pd.to_numeric(w, errors="coerce")
    x = pd.to_numeric(indic, errors="coerce")
    sw = w.sum()
    return float((x * w).sum() / sw * 100) if sw > 0 else np.nan

def wavg(x, w):
    w = pd.to_numeric(w, errors="coerce")
    x = pd.to_numeric(x, errors="coerce")
    sw = w.sum()
    return float((x * w).sum() / sw) if sw > 0 else np.nan

def preparar_lineas_nacionales(path_csv: str) -> pd.DataFrame:
    lp = norm_cols(read_csv_safe(path_csv))
    for c in ["anio", "monto"]:
        if c in lp.columns:
            lp[c] = pd.to_numeric(lp[c], errors="coerce")
    lp["ambito"] = lp["ambito"].str.strip().str.lower()
    lp["tipo"]   = lp["tipo"].str.strip().str.lower()

    # Solo enero (o filas sin mes explícito)
    if "mes" in lp.columns:
        lp = lp[(lp["mes"].str.strip().str.lower() == "ene") | (lp["mes"].isna())]

    lp = lp[lp["anio"].notna() & lp["monto"].notna()]
    piv = lp.pivot_table(index=["anio","ambito"], columns="tipo", values="monto", aggfunc="first").reset_index()
    if "pobreza" in piv.columns:
        piv.rename(columns={"pobreza": "linea_pobreza"}, inplace=True)
    if "extrema" in piv.columns:
        piv.rename(columns={"extrema": "linea_pobreza_extrema"}, inplace=True)
    piv["anio"] = piv["anio"].astype(int)
    return piv[["anio","ambito","linea_pobreza","linea_pobreza_extrema"]]

# Carga
hogares   = norm_cols(read_csv_safe(PATH_HOGARES))
poblacion = norm_cols(read_csv_safe(PATH_POBLACION))
trabajos  = norm_cols(read_csv_safe(PATH_TRABAJOS))
viviendas = norm_cols(read_csv_safe(PATH_VIVIENDAS)) if os.path.exists(PATH_VIVIENDAS) else pd.DataFrame()
sinco     = norm_cols(read_csv_safe(PATH_SINCO))

# Validaciones mínimas (llaves y factor)
for name, df in [("hogares", hogares), ("poblacion", poblacion), ("trabajos", trabajos)]:
    for k in ["folioviv", "foliohog"]:
        if k not in df.columns:
            raise KeyError(f"Falta la llave '{k}' en {name}.")
if "numren" not in poblacion.columns or "numren" not in trabajos.columns:
    raise KeyError("Falta 'numren' en POBLACION/TRABAJOS.")
if "factor" not in poblacion.columns and "factor" not in trabajos.columns:
    raise KeyError("Se espera 'factor' (peso) en POBLACION o TRABAJOS.")

# cast pesos
for df in (hogares, poblacion, trabajos):
    if "factor" in df.columns:
        df["factor"] = pd.to_numeric(df["factor"], errors="coerce")

# Ocupación cultural (sinco + jefes)
if "sinco" not in trabajos.columns:
    cand = [c for c in trabajos.columns if re.search(r"\bsinco\b|ocup", c)]
    if not cand:
        raise KeyError("No se encontró columna 'sinco' ni alternativa en TRABAJOS.")
    trabajos.rename(columns={cand[0]: "sinco"}, inplace=True)

if not {"clave", "cultural"}.issubset(sinco.columns):
    raise KeyError("sinco_interes_cultural.csv debe tener columnas 'clave' y 'cultural'.")

trabajos["sinco"] = trabajos["sinco"].astype(str).str.strip()
sinco["clave"]    = sinco["clave"].astype(str).str.strip()
sinco["cultural"] = sinco["cultural"].astype(str).str.strip()

trab_cultura = trabajos.merge(
    sinco[["clave","cultural"]],
    left_on="sinco", right_on="clave", how="left"
)
trab_cultura["_es_cultura"] = trab_cultura["cultural"] == "1"

# Jefes de hogar (parentesco 101)
CODIGO_JEFE = {"101"}
poblacion["_es_jefe"] = poblacion["parentesco"].astype(str).str.zfill(3).isin(CODIGO_JEFE)

# Unión TRABAJOSxPOBLACION y coalesce de pesos
cj = trab_cultura.merge(
    poblacion[["folioviv","foliohog","numren","_es_jefe","entidad"] + (["factor"] if "factor" in poblacion.columns else [])],
    on=["folioviv","foliohog","numren"], how="left"
)
fac_cols = [c for c in cj.columns if re.fullmatch(r"factor(_x|_y)?", c)]
if not fac_cols:
    raise KeyError("Después del merge no se encontró ninguna columna de factor en 'cj'.")
cj["factor_w"] = pd.to_numeric(cj[fac_cols].bfill(axis=1).iloc[:, 0], errors="coerce")

# Filtrar jefes + cultura
cj = cj[(cj["_es_cultura"]) & (cj["_es_jefe"])].copy()
if cj.empty:
    raise ValueError("No se encontraron jefes de hogar con ocupaciones culturales.")

# Ámbito (si no viene, se deriva de folioviv)
if "ambito" in poblacion.columns:
    amb = poblacion[["folioviv","foliohog","ambito"]].drop_duplicates()
    amb["ambito"] = amb["ambito"].str.strip().str.lower()
    cj = cj.merge(amb, on=["folioviv","foliohog"], how="left")
if "ambito" not in cj.columns or cj["ambito"].isna().any():
    cj["ambito"] = cj["folioviv"].astype(str).map(ambito_from_folioviv)

# Tamaño del hogar y dependientes
tam_hogar = (
    poblacion.groupby(["folioviv","foliohog"], dropna=False)["numren"]
    .count().reset_index().rename(columns={"numren":"tam_hogar"})
)
cj = cj.merge(tam_hogar, on=["folioviv","foliohog"], how="left")
cj["tam_hogar"]    = pd.to_numeric(cj["tam_hogar"], errors="coerce")
cj["dependientes"] = (cj["tam_hogar"] - 1).clip(lower=0)

# Ingreso corriente per cápita (mensual)
if not os.path.exists(PATH_CONC):
    raise FileNotFoundError("Falta data/concentradohogar.csv (necesito ingresos del hogar).")
conc = norm_cols(read_csv_safe(PATH_CONC))

# Detectar columna de ingreso corriente (trimestral o mensual)
col_tri = next((c for c in conc.columns if re.search(r"ing.*cor.*tri", c)), None)
col_men = next((c for c in conc.columns if re.search(r"^ing.*cor(?!.*tri)", c)), None)
col_ing = col_tri or col_men
if col_ing is None:
    raise KeyError("No se encontró ingreso corriente en CONCENTRADOHOGAR (ej. 'ing_cor_tri' o 'ing_cor').")

conc[col_ing] = pd.to_numeric(conc[col_ing], errors="coerce")
conc = conc.merge(tam_hogar, on=["folioviv","foliohog"], how="left")
conc["tam_hogar"] = pd.to_numeric(conc["tam_hogar"], errors="coerce")

# Trimestral → mensual; Mensual → directo
if col_tri:
    conc["ingreso_pc_mensual"] = (conc[col_ing] / 3.0) / conc["tam_hogar"]
else:
    conc["ingreso_pc_mensual"] = conc[col_ing] / conc["tam_hogar"]

# Agregar a cj
cj = cj.merge(conc[["folioviv","foliohog","ingreso_pc_mensual"]], on=["folioviv","foliohog"], how="left")

# Líneas de pobreza
if not os.path.exists(PATH_LINEAS):
    raise FileNotFoundError("Falta data/lineas_pobreza_coneval.csv.")

lineas = preparar_lineas_nacionales(PATH_LINEAS)
lp_2025 = lineas[lineas["anio"] == int(ANIO_LINEAS)].copy()
if lp_2025.empty:
    raise ValueError("No encontré líneas para 2025 en lineas_pobreza_coneval.csv (columna 'anio').")

# Empatar por ámbito (urbano/rural)
cj = cj.merge(lp_2025[["ambito","linea_pobreza","linea_pobreza_extrema"]], on="ambito", how="left")

# Banderas: insuficiencia de ingreso (dimensión de bienestar)
cj["_bajo_lp"]   = (cj["ingreso_pc_mensual"] < cj["linea_pobreza"]).astype(float)
cj["_bajo_lpei"] = (cj["ingreso_pc_mensual"] < cj["linea_pobreza_extrema"]).astype(float)

# Tabulados
frames = []

# Nacional
tot_w = cj["factor_w"].sum()
df_nat = pd.DataFrame({
    "nivel": ["Nacional"],
    "grupo": ["Total"],
    "personas_ocup_cultura_jefes": [tot_w],
    "tam_hogar_promedio": [wavg(cj["tam_hogar"], cj["factor_w"])],
    "dependientes_promedio": [wavg(cj["dependientes"], cj["factor_w"])],
    "ingreso_pc_mensual_prom": [wavg(cj["ingreso_pc_mensual"], cj["factor_w"])],
    "bajo_lp_%": [wmean(cj["_bajo_lp"], cj["factor_w"])],
    "bajo_lpei_%": [wmean(cj["_bajo_lpei"], cj["factor_w"])],
})
frames.append(df_nat)

# Por entidad (si está)
if "entidad" in cj.columns:
    g = cj.groupby("entidad", dropna=False)
    df_ent = pd.DataFrame({
        "nivel": "Por entidad",
        "grupo": g.apply(lambda d: d.name),
        "personas_ocup_cultura_jefes": g["factor_w"].sum().values,
        "tam_hogar_promedio": g.apply(lambda d: wavg(d["tam_hogar"], d["factor_w"])).values,
        "dependientes_promedio": g.apply(lambda d: wavg(d["dependientes"], d["factor_w"])).values,
        "ingreso_pc_mensual_prom": g.apply(lambda d: wavg(d["ingreso_pc_mensual"], d["factor_w"])).values,
        "bajo_lp_%": g.apply(lambda d: wmean(d["_bajo_lp"], d["factor_w"])).values,
        "bajo_lpei_%": g.apply(lambda d: wmean(d["_bajo_lpei"], d["factor_w"])).values,
    })
    frames.append(df_ent)

# Por ámbito
g = cj.groupby("ambito", dropna=False)
df_amb = pd.DataFrame({
    "nivel": "Por ámbito",
    "grupo": g.apply(lambda d: d.name),
    "personas_ocup_cultura_jefes": g["factor_w"].sum().values,
    "tam_hogar_promedio": g.apply(lambda d: wavg(d["tam_hogar"], d["factor_w"])).values,
    "dependientes_promedio": g.apply(lambda d: wavg(d["dependientes"], d["factor_w"])).values,
    "ingreso_pc_mensual_prom": g.apply(lambda d: wavg(d["ingreso_pc_mensual"], d["factor_w"])).values,
    "bajo_lp_%": g.apply(lambda d: wmean(d["_bajo_lp"], d["factor_w"])).values,
    "bajo_lpei_%": g.apply(lambda d: wmean(d["_bajo_lpei"], d["factor_w"])).values,
})
frames.append(df_amb)

# Por entidad y ámbito (opcional)
if "entidad" in cj.columns:
    g = cj.groupby(["entidad","ambito"], dropna=False)
    df_ea = pd.DataFrame({
        "nivel": "Por entidad y ámbito",
        "grupo": g.apply(lambda d: f"{d.name[0]} - {d.name[1]}"),
        "personas_ocup_cultura_jefes": g["factor_w"].sum().values,
        "tam_hogar_promedio": g.apply(lambda d: wavg(d["tam_hogar"], d["factor_w"])).values,
        "dependientes_promedio": g.apply(lambda d: wavg(d["dependientes"], d["factor_w"])).values,
        "ingreso_pc_mensual_prom": g.apply(lambda d: wavg(d["ingreso_pc_mensual"], d["factor_w"])).values,
        "bajo_lp_%": g.apply(lambda d: wmean(d["_bajo_lp"], d["factor_w"])).values,
        "bajo_lpei_%": g.apply(lambda d: wmean(d["_bajo_lpei"], d["factor_w"])).values,
    })
    frames.append(df_ea)

resumen = pd.concat(frames, ignore_index=True)

# =========================
# SALIDAS
# =========================
resumen_path = os.path.join(OUT_DIR, "resumen_cultura_jefes_ingresos.csv")
micro_path   = os.path.join(OUT_DIR, "micro_cultura_jefes_ingresos.csv")

resumen.to_csv(resumen_path, index=False, encoding="utf-8")

cols_micro = [
    "folioviv","foliohog","numren","entidad","ambito","sinco","factor_w",
    "tam_hogar","dependientes","ingreso_pc_mensual",
    "linea_pobreza","linea_pobreza_extrema","_bajo_lp","_bajo_lpei"
]
cols_micro = [c for c in cols_micro if c in cj.columns]
cj[cols_micro].to_csv(micro_path, index=False, encoding="utf-8")

print("Archivos generados:")
print(" -", resumen_path)
print(" -", micro_path)