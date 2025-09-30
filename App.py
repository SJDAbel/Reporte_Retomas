# app.py
# --- Streamlit BI para Adelanta Factoring ---
# Seguridad + 2 pestañas + Postgres + auto-actualización 2 veces/día + refresh manual
# Autor: ChatGPT (Plan BI) – listo para adaptar a tu proyecto

from __future__ import annotations
import os
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import streamlit as st

import gspread
from google.oauth2.service_account import Credentials

from sqlalchemy import create_engine

# Charts
from streamlit_echarts import st_echarts

# Seguridad
import streamlit_authenticator as stauth

# DB
from sqlalchemy import create_engine, text

# =============================
# CONFIGURACIÓN & SEGURIDAD
# =============================
st.set_page_config(page_title="Adelanta BI – Comercial", layout="wide", page_icon="📊")

# =============================
# AUTH – streamlit-authenticator
# =============================

def _build_auth():
        
    secrets = dict(st.secrets)
    cred_list = secrets.get("credentials", {}).get("users", [])
    auth_cfg  = secrets.get("auth", {})

    # Validaciones mínimas
    if not cred_list:
        st.error("No hay usuarios en [credentials].users")
        st.stop()
        
    creds = {
        "usernames": {
            u["username"]: {
                "name": u.get("name", u["username"]),
                "email": u.get("email", f'{u["username"]}@example.local'),  # <-- obligatoria
                "password": u["password"],
                # opcional: si más adelante usas roles
                "roles": u.get("roles", []),
            }
            for u in cred_list
        }
    }

    cookie_name = auth_cfg.get("cookie_name", "adelanta_bi_cookie")
    cookie_key  = auth_cfg.get("cookie_key")
    cookie_days = auth_cfg.get("cookie_expiry_days", 7)
    if not cookie_key:
        st.error("Falta [auth].cookie_key en secrets.toml (usa una cadena fija).")
        st.stop()

    return stauth.Authenticate(
        credentials=creds,
        cookie_name=cookie_name,
        key=cookie_key,
        cookie_expiry_days=cookie_days,
        preauthorized=[],
    )


authenticator = _build_auth()

login_fields = {
    "Form name": "Iniciar sesión",
    "Username": "Usuario",
    "Password": "Contraseña",
    "Login": "Ingresar",
}

# --- contenedor que mantiene el formulario visible en reruns ---
placeholder = st.empty()

with placeholder.container():
    # API 0.4.x: pinta el formulario; NO desempaquetes
    authenticator.login(fields=login_fields, location="main")
    
auth_status = st.session_state.get("authentication_status", None)
username    = st.session_state.get("username", None)
name        = st.session_state.get("name", None)


if auth_status is None:
        st.info("Ingresa tus credenciales para continuar.")
        st.stop()  # <- detiene la app pero mantiene el contenido del 'placeholder'

if auth_status is False:
        st.error("Usuario o contraseña incorrectos.")
        st.stop()

with st.sidebar:
    st.caption(f"Usuario: {name or username}")
    authenticator.logout("Cerrar sesión", "sidebar")


# =============================
# DB & CACHING
# =============================

@st.cache_resource(show_spinner=False)
def get_engine():
    url = st.secrets.get("db", {}).get("url")
    if not url:
        st.error("Configura [db].url en secrets.")
        st.stop()
    try:
        engine = create_engine(url, pool_pre_ping=True)
        return engine
    except Exception as e:
        st.error(f"Error creando engine: {e}")
        st.stop()


@st.cache_data(show_spinner=True, ttl=60*60*12)
def load_data(fi: Optional[date]=None, ff: Optional[date]=None, ejecutivos: Optional[List[str]]=None) -> pd.DataFrame:

    table = st.secrets.get("db", {}).get("table", "kpi_acumulado")
    engine = get_engine()

    # 1) Descubrir columnas disponibles en la tabla/vista
    with engine.begin() as conn:
        res = conn.execute(text(f'SELECT * FROM {table} LIMIT 0'))
        available_cols = list(res.keys())

    # 2) Columnas que nos gustaría traer
    desired = [
        "FechaOperacion","FechaConfirmado","NetoConfirmado","Moneda",
        "TipoPago","TipoOperacion","Ejecutivo",
        "RUCCliente","RazonSocialCliente",
        "RUCPagador","RazonSocialPagador"
    ]

    # 3) Intersección: solo pedimos lo que realmente existe
    select_cols = [c for c in desired if c in available_cols]
    select_sql = ", ".join(f'"{c}"' for c in select_cols)

    # 4) WHERE dinámico
    where = []
    params = {}
    if fi:
        where.append('("FechaOperacion" >= :fi OR "FechaConfirmado" >= :fi)')
        params["fi"] = datetime.combine(fi, datetime.min.time())
    if ff:
        where.append('("FechaOperacion" <= :ff OR "FechaConfirmado" <= :ff)')
        params["ff"] = datetime.combine(ff, datetime.max.time())
    if ejecutivos:
        if "Ejecutivo" in available_cols:
            where.append('"Ejecutivo" = ANY(:ejes)')
            params["ejes"] = ejecutivos

    where_sql = (" WHERE " + " AND ".join(where)) if where else ""

    # 5) Ejecutar SELECT (sin coma final 😉)
    sql = f"""
        SELECT
            {select_sql}
        FROM {table}
        {where_sql}
    """

    with engine.begin() as conn:
        df = pd.read_sql_query(text(sql), conn, params=params)

    # 6) Tipos y columnas faltantes
    for c in ["FechaOperacion","FechaConfirmado"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    if "NetoConfirmado" in df.columns:
        df["NetoConfirmado"] = pd.to_numeric(df["NetoConfirmado"], errors="coerce")
    if "Moneda" in df.columns:
        df["Moneda"] = df["Moneda"].astype(str)

    return df


# =============================
# GSHEETS – SECTORES / GRUPO ECO
# =============================

@st.cache_data(show_spinner=False, ttl=60*60*6)
def load_sector_map(
    spreadsheet_id: str | None = None,
    worksheet: str | None = None,
) -> pd.DataFrame:
    
    # 1) Parámetros desde secrets si no se pasan
    spreadsheet_id = spreadsheet_id or st.secrets["gsheets"]["spreadsheet_id"]
    worksheet      = worksheet      or st.secrets["gsheets"]["worksheet"]

    # 2) Autenticación (solo lectura)
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets.readonly",
        "https://www.googleapis.com/auth/drive.readonly",
    ]
    creds = Credentials.from_service_account_info(
        dict(st.secrets["gcp_service_account"]),
        scopes=scopes
    )
    gc = gspread.authorize(creds)

    # 3) Leer datos crudos
    ws = gc.open_by_key(spreadsheet_id).worksheet(worksheet)
    records = ws.get_all_records()  # list[dict]

    if not records:
        # DataFrame vacío pero con columnas esperadas para no romper merges
        return pd.DataFrame(columns=["RUCPagador", "SectorEconomico", "GrupoEconomico"])

    df_sec = pd.DataFrame(records)

    # 4) Localizar columnas de forma flexible
    cols_lower = {c.lower().strip(): c for c in df_sec.columns}

    def pick(*cands: str) -> str | None:
        for c in cands:
            hit = cols_lower.get(c.lower().strip())
            if hit:
                return hit
        return None

    c_ruc     = pick("RUC")
    c_sector  = pick("SECTOR")
    c_grupo   = pick("GRUPO ECO.")

    out = df_sec.rename(
        columns= {
            c_ruc: "RUCPagador",
            c_sector: "SectorEconomico",
            c_grupo: "GrupoEconomico",
        }
    )

    out = out[[
        "RUCPagador",
        "SectorEconomico",
        "GrupoEconomico"
    ]].copy()
        
    # 7) Filtrar filas sin RUC y deduplicar por RUCPagador
    out = out.dropna(subset=["RUCPagador"])
    out = out.drop_duplicates(subset=["RUCPagador"], keep="first").reset_index(drop=True)
    
    # 8) Devolver solo columnas estándar (las necesarias para el merge)
    return out

def normalize_ruc(s: pd.Series) -> pd.Series:
    # Devuelve siempre string de 11 dígitos
    return (
        s.astype("string")              # dtype string, no object
         .fillna("")
         .str.replace(r"\.0$", "", regex=True)   # si venía como 12345678901.0
         .str.replace(r"\D", "", regex=True)     # deja solo dígitos
         .str[-11:]                              
         .str.zfill(11)
    )


def enrich_with_sector_map(df: pd.DataFrame, sec: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.assign(SectorEconomico="SIN SECTOR")

    if "RUCPagador" not in df.columns:
        st.warning("df no tiene RUCPagador; no se puede enriquecer sectores.")
        return df.assign(SectorEconomico="SIN SECTOR")

    d = df.copy()

    # 🔑 NORMALIZA Y FIJA DTYPE A "string" EN AMBOS LADOS
    d["RUCPagador"] = normalize_ruc(d["RUCPagador"]).astype("string")

    if sec is None or sec.empty or "RUCPagador" not in sec.columns:
        d["SectorEconomico"] = d.get("SectorEconomico", pd.NA).fillna("SIN SECTOR")
        return d

    sec2 = sec.copy()
    sec2["RUCPagador"] = normalize_ruc(sec2["RUCPagador"]).astype("string")
    if "SectorEconomico" in sec2.columns:
        sec2["SectorEconomico"] = sec2["SectorEconomico"].astype("string").str.strip()
    else:
        sec2["SectorEconomico"] = pd.NA

    # Merge con diagnóstico
    d = d.merge(
        sec2[["RUCPagador", "SectorEconomico"]],
        on="RUCPagador", how="left", suffixes=("", "_GS"), indicator=True
    )

    if "SectorEconomico_GS" in d.columns:
        use_gs = d["SectorEconomico_GS"].notna() & d["SectorEconomico_GS"].str.strip().ne("")
        d["SectorEconomico"] = np.where(use_gs, d["SectorEconomico_GS"], d.get("SectorEconomico"))
        d.drop(columns=["SectorEconomico_GS"], inplace=True, errors="ignore")

    d["SectorEconomico"] = d["SectorEconomico"].fillna("SIN SECTOR")

    try:
        total = len(d)
        matched = int((d["_merge"] == "both").sum())
        left_only = int((d["_merge"] == "left_only").sum())
        st.caption(f"🧭 Sectores: {matched}/{total} filas matcheadas por RUC; sin match: {left_only}.")
    finally:
        d.drop(columns=["_merge"], inplace=True, errors="ignore")

    return d




# =============================
# HELPERS DE MONTOS & TABLAS
# =============================

def add_monto_pen(df: pd.DataFrame, tc: float) -> pd.DataFrame:
    if df.empty:
        return df
    d = df.copy()
    d["NetoConfirmado"] = pd.to_numeric(d["NetoConfirmado"], errors="coerce")
    factor = d["Moneda"].map({"PEN": 1.0, "USD": float(tc)}).fillna(1.0)
    d["MontoNetoPEN"] = d["NetoConfirmado"] * factor
    return d


def fmt_money(v) -> str:
    try:
        return f"{float(v):,.2f}"
    except Exception:
        return "-"


def rango_coloc_cobro(df: pd.DataFrame, tc: float, fi, ff) -> pd.DataFrame:
    """Diario por rango: Fecha | Cobro | Coloco | Gap."""
    if df.empty:
        return pd.DataFrame(columns=["Fecha","Cobro","Coloco","Gap"])
    d = add_monto_pen(df, tc)
    d["FechaOperacion"]  = pd.to_datetime(d["FechaOperacion"], errors="coerce")
    d["FechaConfirmado"] = pd.to_datetime(d["FechaConfirmado"], errors="coerce")
    fi = pd.to_datetime(fi).normalize(); ff = pd.to_datetime(ff).normalize()

    coloc_mask = (
        d["FechaOperacion"].between(fi, ff, inclusive="both")
    )
    g_col = (d.loc[coloc_mask, ["FechaOperacion","MontoNetoPEN"]]
               .assign(Fecha=lambda x: x["FechaOperacion"].dt.normalize())
               .groupby("Fecha", as_index=False)["MontoNetoPEN"].sum()
               .rename(columns={"MontoNetoPEN":"Coloco"}))

    cobro_mask = (
        d["FechaConfirmado"].between(fi, ff, inclusive="both")
    )
    g_cob = (d.loc[cobro_mask, ["FechaConfirmado","MontoNetoPEN"]]
               .assign(Fecha=lambda x: x["FechaConfirmado"].dt.normalize())
               .groupby("Fecha", as_index=False)["MontoNetoPEN"].sum()
               .rename(columns={"MontoNetoPEN":"Cobro"}))

    out = pd.merge(g_cob, g_col, on="Fecha", how="outer").fillna(0.0)
    all_days = pd.DataFrame({"Fecha": pd.date_range(fi, ff, freq="D")})
    out = all_days.merge(out, on="Fecha", how="left").fillna(0.0)
    out["Gap"] = out["Coloco"] - out["Cobro"]
    return out.sort_values("Fecha").reset_index(drop=True)


def empresas_en_rango(df: pd.DataFrame, tc: float, fi, ff, por="auto") -> pd.DataFrame:
    """Empresas que colocaron o cobraron en el rango (formato amigable)."""
    if df.empty:
        return df.iloc[0:0]
    d = add_monto_pen(df, tc)
    d["FechaOperacion"]  = pd.to_datetime(d["FechaOperacion"], errors="coerce")
    d["FechaConfirmado"] = pd.to_datetime(d["FechaConfirmado"], errors="coerce")
    fi = pd.to_datetime(fi); ff = pd.to_datetime(ff)

    es_coloc = (
        
        d["FechaOperacion"].between(fi, ff, inclusive="both")
    )
    es_cobro = (
        d["TipoPago"].astype(str).str.strip().ne("") &
        d["FechaConfirmado"].between(fi, ff, inclusive="both")
    )

    if por == "cliente" or (por=="auto" and "RUCCliente" in d.columns):
        id_cols = ["RUCCliente","RazonSocialCliente"]
    else:
        id_cols = ["RUCPagador","RazonSocialPagador"]

    g_col = (d.loc[es_coloc, id_cols + ["MontoNetoPEN"]]
               .groupby(id_cols, as_index=False)["MontoNetoPEN"].sum()
               .rename(columns={"MontoNetoPEN":"ColocoPEN"}))
    g_cob = (d.loc[es_cobro, id_cols + ["MontoNetoPEN"]]
               .groupby(id_cols, as_index=False)["MontoNetoPEN"].sum()
               .rename(columns={"MontoNetoPEN":"CobroPEN"}))

    emp = g_col.merge(g_cob, on=id_cols, how="outer").fillna(0.0)
    emp["GapPEN"] = emp["ColocoPEN"] - emp["CobroPEN"]

    view = emp.copy()
    view["Coloco"] = view["ColocoPEN"].map(fmt_money)
    view["Cobro"]  = view["CobroPEN"].map(fmt_money)
    view["Gap"]    = view["GapPEN"].map(fmt_money)

    nice_cols = id_cols + ["Coloco","Cobro","Gap","ColocoPEN","CobroPEN","GapPEN"]
    return view.loc[:, nice_cols].sort_values(["GapPEN","CobroPEN"], ascending=[True, False]).reset_index(drop=True)


# =============================
# SIDEBAR – FILTROS & REFRESH
# =============================


with st.sidebar:
    st.header("⚙️ Controles")

    # Actualización manual (limpia cache_data)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔄 Datos", use_container_width=True):
            load_data.clear()
            st.session_state["last_update_db"] = datetime.now()
            st.rerun()
    with c2:
        if st.button("🧭 Sector", use_container_width=True):
            load_sector_map.clear()
            st.session_state["last_update_gs"] = datetime.now()
            st.rerun()

    # Mostrar info en el sidebar
    st.sidebar.markdown("### ⏱️ Última actualización")
    if "last_update_db" in st.session_state:
        st.sidebar.caption(f"DB: {st.session_state['last_update_db'].strftime('%d/%m/%Y %H:%M:%S')}")
    if "last_update_gs" in st.session_state:
        st.sidebar.caption(f"Google: {st.session_state['last_update_gs'].strftime('%d/%m/%Y %H:%M:%S')}")
    
    tc = st.sidebar.number_input("TC Compra USD→PEN", min_value=0.0001, value=3.75, step=0.01,
                                         help="Tipo de cambio aplicado para conversión de dólares (USD) a soles (PEN).")
    
    st.caption(f"Streamlit: {st.__version__} | NumPy: {np.__version__} | Pandas: {pd.__version__}")

# =============================
# PESTAÑA 1 – Colocación vs Cobrado
# =============================

st.title("📊 Adelanta BI – Reporte de Retomas")
tab1 = st.tabs(["Colocación y Cobranza"])[0] 

with tab1:
    st.subheader("Análisis de Saldos Diarios")

    # Rango de fechas disponible a partir de DB (consulta rápida de min/max)
    # Para simplificar, cargamos sin filtros y luego proponemos rango
    
    df_base = load_data()

    if df_base.empty:
        st.info("No hay datos disponibles.")
        st.stop()

    fechas_disp = pd.concat([
        df_base["FechaOperacion"].dropna(),
        df_base["FechaConfirmado"].dropna(),
    ])
    
    fechas_disp = pd.to_datetime(fechas_disp, errors="coerce").dropna()
    fmin, fmax = fechas_disp.min().date(), fechas_disp.max().date()

    colA, colB = st.columns([1,1])
    with colA:
        fi_ff = st.date_input("Rango de fechas", value=(fmax - timedelta(days=30), fmax), min_value=fmin, max_value=fmax)
        if isinstance(fi_ff, tuple) and len(fi_ff)==2:
            fi_sel, ff_sel = fi_ff
        else:
            fi_sel = ff_sel = fi_ff
    with colB:
        ejes = sorted(df_base["Ejecutivo"].dropna().unique().tolist())
        eje_sel = st.multiselect("Filtrar por Ejecutivo(s)", options=ejes, default=None)

    # Cargar ya filtrado desde DB (aprovecha índices) → evita procesar demás
    df = load_data(fi_sel, ff_sel, eje_sel if eje_sel else None)

    # Tabla 1: Diario por fecha (Cobro vs Coloco)
    df_rango = rango_coloc_cobro(df, tc, fi_sel, ff_sel)
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.metric("Saldo Cobrado (PEN)",  fmt_money(df_rango["Cobro"].sum()))
    c2.metric("Saldo Colocado (PEN)", fmt_money(df_rango["Coloco"].sum()))
    c3.metric("Gap",          fmt_money(df_rango["Gap"].sum()))
    
    show = df_rango.copy()
    show["Fecha"]  = show["Fecha"].dt.date
    show["Cobro"]  = show["Cobro"].map(fmt_money)
    show["Coloco"] = show["Coloco"].map(fmt_money)
    show["Gap"]    = show["Gap"].map(fmt_money)

    st.dataframe(show, use_container_width=True, hide_index=True, height=360)

    # Tabla 2: Empresas que cobraron / desembolsaron en el rango
    st.markdown("---")
    st.markdown("### Detalle de Empresas (colocó/cobró)")
    emp = empresas_en_rango(df, tc, fi_sel, ff_sel, por="auto")
    st.dataframe(emp.drop(columns=["ColocoPEN","CobroPEN","GapPEN"]), use_container_width=True, hide_index=True, height=420)

