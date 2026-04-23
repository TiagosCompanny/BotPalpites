import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from pathlib import Path
import pytz

BASE_DIR = Path(__file__).resolve().parent
ARQUIVO_BASE = BASE_DIR.parent / "DadosPassagens" / "dados_todas_passagens.xlsx"

# 📍 Coordenada fixa (Japão)
LATITUDE_FIXA = 35.6938
LONGITUDE_FIXA = 139.7005

COLUNAS_CLIMA = [
    "temperatura_2m",
    "umidade_relativa",
    "chuva_mm",
    "cobertura_nuvens",
    "estava_chovendo",
]


def criar_sessao():
    session = requests.Session()

    retry = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )

    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session


session = criar_sessao()


def garantir_colunas_clima(df):
    for coluna in COLUNAS_CLIMA:
        if coluna not in df.columns:
            df[coluna] = pd.NA
    return df


def buscar_clima_periodo(latitude, longitude, data_inicial, data_final):
    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": data_inicial.strftime("%Y-%m-%d"),
        "end_date": data_final.strftime("%Y-%m-%d"),
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,cloud_cover",
        "timezone": "Asia/Tokyo",
    }

    response = session.get(url, params=params, timeout=60)
    response.raise_for_status()
    dados = response.json()

    hourly = dados.get("hourly", {})

    df_clima = pd.DataFrame(
        {
            "data_hora_hora": pd.to_datetime(hourly.get("time", []), errors="coerce"),
            "temperatura_2m": hourly.get("temperature_2m", []),
            "umidade_relativa": hourly.get("relative_humidity_2m", []),
            "chuva_mm": hourly.get("precipitation", []),
            "cobertura_nuvens": hourly.get("cloud_cover", []),
        }
    )

    if not df_clima.empty:
        df_clima["estava_chovendo"] = df_clima["chuva_mm"].apply(
            lambda x: 1 if pd.notna(x) and x > 0 else 0
        )

    return df_clima


def atualizar_dados_climaticos(caminho_arquivo, coluna_data="data"):
    if not os.path.exists(caminho_arquivo):
        print(f"Arquivo não encontrado: {caminho_arquivo}")
        return

    df = pd.read_excel(caminho_arquivo)
    print(f"Linhas carregadas: {len(df)}")

    if df.empty:
        print("Planilha vazia.")
        return

    df = garantir_colunas_clima(df)

    # 🔥 CONVERSÃO DE DATAS
    df[coluna_data] = pd.to_datetime(df[coluna_data], errors="coerce")

    df["data_br"] = df[coluna_data].dt.tz_localize("America/Sao_Paulo")
    df["data_japao"] = df["data_br"].dt.tz_convert("Asia/Tokyo")

    df["data_hora_hora"] = df["data_japao"].dt.floor("h").dt.tz_localize(None)

    df["latitude"] = LATITUDE_FIXA
    df["longitude"] = LONGITUDE_FIXA

    df_pendentes = df[df["temperatura_2m"].isna()].copy()
    print(f"Linhas com clima pendente: {len(df_pendentes)}")

    if df_pendentes.empty:
        print("Nada para atualizar.")
        return

    data_min = df_pendentes["data_hora_hora"].min()
    data_max = df_pendentes["data_hora_hora"].max()

    # 🔥 BLOQUEIO DE DATAS FUTURAS
    tz_japao = pytz.timezone("Asia/Tokyo")
    agora_japao = datetime.now(tz_japao)

    data_limite = (agora_japao - timedelta(days=1)).replace(
        hour=23, minute=59, second=59, microsecond=0
    ).replace(tzinfo=None)

    print(f"Data limite API (histórico): {data_limite}")

    # Se tudo for futuro → aborta
    if data_min > data_limite:
        print("⚠️ Todas as datas são futuras. Nada para buscar no histórico.")
        return

    # Ajusta o máximo permitido
    data_max = min(data_max, data_limite)

    print(f"Consultando clima de {data_min} até {data_max}")

    try:
        df_clima = buscar_clima_periodo(
            LATITUDE_FIXA, LONGITUDE_FIXA, data_min, data_max
        )

        if df_clima.empty:
            print("Sem retorno de clima.")
            return

        df = df.merge(
            df_clima,
            on="data_hora_hora",
            how="left",
            suffixes=("_antigo", ""),
        )

        colunas_para_remover = [
            f"{c}_antigo" for c in COLUNAS_CLIMA if f"{c}_antigo" in df.columns
        ]

        df = df.drop(
            columns=colunas_para_remover + ["data_hora_hora", "data_br", "data_japao"],
            errors="ignore",
        )

        df.to_excel(caminho_arquivo, index=False)

        print("✅ Planilha atualizada com sucesso!")

    except Exception as e:
        print(f"❌ Erro ao buscar clima: {e}")


# EXECUÇÃO
atualizar_dados_climaticos(ARQUIVO_BASE, coluna_data="abertura")