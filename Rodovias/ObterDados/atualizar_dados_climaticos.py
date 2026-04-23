import os
import time
import requests
import pandas as pd
from pathlib import Path
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_DIR = Path(__file__).resolve().parent
ARQUIVO_BASE = BASE_DIR.parent / "DadosRodovias" / "dados_todas_rodovias.xlsx"

coordenadas_rodovias = {
    ("Rodovia Arão Sahm", 95): {"latitude": -22.9256, "longitude": -46.5529},
    ("Doutor Manoel Hyppolito Rego", 83): {"latitude": -23.5663, "longitude": -45.2793},
    ("Doutor Manoel Hyppolito Rego", 83.5): {"latitude": -23.5663, "longitude": -45.2793},
    ("Doutor Manoel Hyppolito Rego", 110): {"latitude": -23.6980, "longitude": -45.4395},
    ("Doutor Manoel Hyppolito Rego", 110.8): {"latitude": -23.6980, "longitude": -45.4395},
    ("Floriano Rodrigues Pinheiro", 46): {"latitude": -22.7561, "longitude": -45.6102},
    ("Floriano Rodrigues Pinheiro", 26): {"latitude": -22.8640, "longitude": -45.6025},
    ("Floriano Rodrigues Pinheiro", 26.5): {"latitude": -22.8640, "longitude": -45.6025},
}

COLUNAS_CLIMA = [
    "temperatura_2m",
    "umidade_relativa",
    "chuva_mm",
    "cobertura_nuvens",
    "estava_chovendo"
]

def criar_sessao():
    session = requests.Session()

    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
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

def adicionar_coordenadas(df):
    if "km" in df.columns:
        df["km"] = pd.to_numeric(df["km"], errors="coerce")

    chaves = list(zip(df["rodovia"], df["km"]))

    df["latitude"] = [
        coordenadas_rodovias.get(chave, {}).get("latitude")
        for chave in chaves
    ]
    df["longitude"] = [
        coordenadas_rodovias.get(chave, {}).get("longitude")
        for chave in chaves
    ]

    return df

def buscar_clima_periodo(latitude, longitude, data_inicial, data_final):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": data_inicial.strftime("%Y-%m-%d"),
        "end_date": data_final.strftime("%Y-%m-%d"),
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,cloud_cover",
        "timezone": "America/Sao_Paulo"
    }

    response = session.get(url, params=params, timeout=60)
    response.raise_for_status()
    dados = response.json()

    hourly = dados.get("hourly", {})

    df_clima = pd.DataFrame({
        "data_hora_hora": pd.to_datetime(hourly.get("time", []), errors="coerce"),
        "temperatura_2m": hourly.get("temperature_2m", []),
        "umidade_relativa": hourly.get("relative_humidity_2m", []),
        "chuva_mm": hourly.get("precipitation", []),
        "cobertura_nuvens": hourly.get("cloud_cover", [])
    })

    if not df_clima.empty:
        df_clima["estava_chovendo"] = df_clima["chuva_mm"].apply(
            lambda x: 1 if pd.notna(x) and x > 0 else 0
        )

    return df_clima

def atualizar_dados_climaticos(caminho_arquivo, coluna_data="abertura"):
    if not os.path.exists(caminho_arquivo):
        print(f"Arquivo não encontrado: {caminho_arquivo}")
        return

    df = pd.read_excel(caminho_arquivo)
    print(f"Linhas carregadas: {len(df)}")

    if df.empty:
        print("Planilha vazia.")
        return

    df = garantir_colunas_clima(df)

    df[coluna_data] = pd.to_datetime(df[coluna_data], errors="coerce")
    df["km"] = pd.to_numeric(df["km"], errors="coerce")

    df = adicionar_coordenadas(df)

    df["data_hora_hora"] = pd.to_datetime(df[coluna_data], errors="coerce").dt.floor("h")

    mascara_clima_faltando = df[COLUNAS_CLIMA].isna().any(axis=1)
    mascara_base_valida = (
        df["latitude"].notna() &
        df["longitude"].notna() &
        df["data_hora_hora"].notna()
    )

    df_pendentes = df[mascara_clima_faltando & mascara_base_valida].copy()

    print(f"Linhas com clima pendente: {len(df_pendentes)}")

    if df_pendentes.empty:
        print("Nenhuma linha pendente de clima.")
        return

    pontos_pendentes = (
        df_pendentes[["latitude", "longitude"]]
        .drop_duplicates()
        .to_dict("records")
    )

    total_atualizadas = 0

    for idx, ponto in enumerate(pontos_pendentes, start=1):
        lat = ponto["latitude"]
        lon = ponto["longitude"]

        idx_ponto = df_pendentes[
            (df_pendentes["latitude"] == lat) &
            (df_pendentes["longitude"] == lon)
        ].index

        df_ponto = df.loc[idx_ponto].copy()

        data_min = df_ponto["data_hora_hora"].min()
        data_max = df_ponto["data_hora_hora"].max()

        print(
            f"Consultando ponto {idx}/{len(pontos_pendentes)} | "
            f"lat={lat} lon={lon} | {data_min} até {data_max}"
        )

        try:
            df_clima = buscar_clima_periodo(
                latitude=lat,
                longitude=lon,
                data_inicial=data_min,
                data_final=data_max
            )

            if df_clima.empty:
                print("Sem retorno de clima para este ponto.")
                continue

            df_merge = df_ponto.merge(
                df_clima,
                on="data_hora_hora",
                how="left",
                suffixes=("", "_novo")
            )

            for coluna in COLUNAS_CLIMA:
                coluna_nova = f"{coluna}_novo"
                if coluna_nova in df_merge.columns:
                    df.loc[idx_ponto, coluna] = df_merge[coluna_nova].values

            total_atualizadas += len(df_ponto)
            time.sleep(1)

        except Exception as e:
            print(f"Erro ao consultar clima do ponto lat={lat}, lon={lon}: {e}")

    if "fechamento" in df.columns:
        df["fechamento"] = pd.to_datetime(df["fechamento"], errors="coerce")
        df = df.sort_values(by="fechamento", ascending=False)

    df = df.drop(columns=["data_hora_hora"], errors="ignore")
    df.to_excel(caminho_arquivo, index=False)

    print(f"Linhas processadas para clima: {total_atualizadas}")
    print(f"Planilha atualizada com sucesso: {caminho_arquivo}")

atualizar_dados_climaticos(ARQUIVO_BASE, coluna_data="abertura")