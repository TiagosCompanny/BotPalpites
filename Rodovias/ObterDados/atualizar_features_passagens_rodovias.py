import os
import numpy as np
import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


BASE_DIR = Path(__file__).resolve().parent
DADOS_DIR = BASE_DIR.parent / "DadosRodovias"

ARQUIVO_BASE = DADOS_DIR / "dados_todas_rodovias.xlsx"
ARQUIVO_PASSAGENS = DADOS_DIR / "passagens_carros_por_mercado.xlsx"


COLUNAS_FEATURES_PASSAGENS = [
    "lag1_pass_total_mercado",
    "lag1_pass_qtd_ultimos_2min",
    "lag1_pass_media_ultimos_2min",
    "lag1_pass_tendencia_5m",
    "roll3_mean_pass_total_mercado",
    "roll5_mean_pass_total_mercado",
    "roll10_mean_pass_total_mercado",
    "lag1_ratio_pass_total_meta",
    "roll3_ratio_pass_total_meta",
    "roll5_ratio_pass_total_meta",
]


def carregar_planilhas():
    if not os.path.exists(ARQUIVO_BASE):
        raise FileNotFoundError(f"Base principal não encontrada: {ARQUIVO_BASE}")

    if not os.path.exists(ARQUIVO_PASSAGENS):
        raise FileNotFoundError(f"Base de passagens não encontrada: {ARQUIVO_PASSAGENS}")

    df_base = pd.read_excel(ARQUIVO_BASE)
    df_passagens = pd.read_excel(ARQUIVO_PASSAGENS)

    return df_base, df_passagens


def preparar_base_principal(df_base):
    df_base = df_base.copy()

    if "id" not in df_base.columns:
        raise ValueError("Coluna 'id' não encontrada na base principal.")

    if "rodovia_identificacao" not in df_base.columns:
        raise ValueError("Coluna 'rodovia_identificacao' não encontrada na base principal.")

    if "abertura" not in df_base.columns:
        raise ValueError("Coluna 'abertura' não encontrada na base principal.")

    if "meta_referencia" not in df_base.columns:
        raise ValueError("Coluna 'meta_referencia' não encontrada na base principal.")

    df_base["id"] = df_base["id"].astype(str)
    df_base["abertura"] = pd.to_datetime(df_base["abertura"], errors="coerce")
    df_base["fechamento"] = pd.to_datetime(df_base["fechamento"], errors="coerce")
    df_base["meta_referencia"] = pd.to_numeric(df_base["meta_referencia"], errors="coerce")

    for coluna in COLUNAS_FEATURES_PASSAGENS:
        if coluna in df_base.columns:
            df_base = df_base.drop(columns=[coluna])

    return df_base


def preparar_base_passagens(df_passagens):
    df_passagens = df_passagens.copy()

    if "MercadoId" not in df_passagens.columns:
        raise ValueError("Coluna 'MercadoId' não encontrada na base de passagens.")

    if "DataHora" not in df_passagens.columns:
        raise ValueError("Coluna 'DataHora' não encontrada na base de passagens.")

    df_passagens["MercadoId"] = df_passagens["MercadoId"].astype(str)
    df_passagens["DataHora"] = pd.to_datetime(df_passagens["DataHora"], errors="coerce")

    df_passagens = df_passagens.dropna(subset=["MercadoId", "DataHora"]).copy()

    return df_passagens


def calcular_resumo_passagens_por_mercado(df_passagens):
    resumo = (
        df_passagens
        .groupby("MercadoId")["DataHora"]
        .agg(
            pass_total_mercado="count",
            primeira_passagem="min"
        )
        .reset_index()
    )

    df_tmp = df_passagens.merge(
        resumo[["MercadoId", "primeira_passagem"]],
        on="MercadoId",
        how="left"
    )

    df_tmp["segundos_desde_inicio"] = (
        df_tmp["DataHora"] - df_tmp["primeira_passagem"]
    ).dt.total_seconds()

    df_tmp["minuto_mercado"] = (
        np.floor(df_tmp["segundos_desde_inicio"] / 60)
        .clip(0, 4)
        .astype(int) + 1
    )

    por_minuto = (
        df_tmp
        .pivot_table(
            index="MercadoId",
            columns="minuto_mercado",
            values="DataHora",
            aggfunc="count",
            fill_value=0
        )
        .reset_index()
    )

    for minuto in [1, 2, 3, 4, 5]:
        if minuto not in por_minuto.columns:
            por_minuto[minuto] = 0

    por_minuto = por_minuto[["MercadoId", 1, 2, 3, 4, 5]]

    por_minuto.columns = [
        "MercadoId",
        "pass_min_1",
        "pass_min_2",
        "pass_min_3",
        "pass_min_4",
        "pass_min_5",
    ]

    resumo = resumo.merge(por_minuto, on="MercadoId", how="left")

    resumo["pass_qtd_primeiros_2min"] = (
        resumo["pass_min_1"] + resumo["pass_min_2"]
    )

    resumo["pass_qtd_ultimos_2min"] = (
        resumo["pass_min_4"] + resumo["pass_min_5"]
    )

    resumo["pass_media_ultimos_2min"] = (
        resumo["pass_qtd_ultimos_2min"] / 2
    )

    resumo["pass_tendencia_5m"] = (
        resumo["pass_qtd_ultimos_2min"] - resumo["pass_qtd_primeiros_2min"]
    )

    return resumo[
        [
            "MercadoId",
            "pass_total_mercado",
            "pass_qtd_ultimos_2min",
            "pass_media_ultimos_2min",
            "pass_tendencia_5m",
        ]
    ]


def aplicar_features_passagens(df_base, resumo_passagens):
    df = df_base.copy()

    df = df.merge(
        resumo_passagens,
        left_on="id",
        right_on="MercadoId",
        how="left"
    )

    if "MercadoId" in df.columns:
        df = df.drop(columns=["MercadoId"])

    df = df.sort_values(
        by=["rodovia_identificacao", "abertura"],
        ascending=[True, True]
    ).reset_index(drop=True)

    grupo = df.groupby("rodovia_identificacao", group_keys=False)

    df["lag1_pass_total_mercado"] = grupo["pass_total_mercado"].shift(1)
    df["lag1_pass_qtd_ultimos_2min"] = grupo["pass_qtd_ultimos_2min"].shift(1)
    df["lag1_pass_media_ultimos_2min"] = grupo["pass_media_ultimos_2min"].shift(1)
    df["lag1_pass_tendencia_5m"] = grupo["pass_tendencia_5m"].shift(1)

    df["roll3_mean_pass_total_mercado"] = grupo["pass_total_mercado"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )

    df["roll5_mean_pass_total_mercado"] = grupo["pass_total_mercado"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )

    df["roll10_mean_pass_total_mercado"] = grupo["pass_total_mercado"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=1).mean()
    )

    meta = df["meta_referencia"].replace(0, np.nan)

    df["lag1_ratio_pass_total_meta"] = (
        df["lag1_pass_total_mercado"] / meta
    )

    df["roll3_ratio_pass_total_meta"] = (
        df["roll3_mean_pass_total_mercado"] / meta
    )

    df["roll5_ratio_pass_total_meta"] = (
        df["roll5_mean_pass_total_mercado"] / meta
    )

    colunas_auxiliares = [
        "pass_total_mercado",
        "pass_qtd_ultimos_2min",
        "pass_media_ultimos_2min",
        "pass_tendencia_5m",
    ]

    df = df.drop(columns=[c for c in colunas_auxiliares if c in df.columns])

    if "fechamento" in df.columns:
        df = df.sort_values(by="fechamento", ascending=False).reset_index(drop=True)

    return df


def atualizar_features_passagens():
    print("Carregando planilhas...")

    df_base, df_passagens = carregar_planilhas()

    print(f"Registros na base principal: {len(df_base)}")
    print(f"Registros na base de passagens: {len(df_passagens)}")

    df_base = preparar_base_principal(df_base)
    df_passagens = preparar_base_passagens(df_passagens)

    print("Calculando resumo de passagens por mercado...")

    resumo_passagens = calcular_resumo_passagens_por_mercado(df_passagens)

    print(f"Mercados com passagens encontradas: {len(resumo_passagens)}")

    print("Aplicando features de passagens passadas...")

    df_atualizado = aplicar_features_passagens(df_base, resumo_passagens)

    df_atualizado.to_excel(ARQUIVO_BASE, index=False)

    print("Base atualizada com sucesso.")
    print(f"Arquivo salvo em: {ARQUIVO_BASE}")

    print("Colunas atualizadas:")

    for coluna in COLUNAS_FEATURES_PASSAGENS:
        preenchidos = df_atualizado[coluna].notna().sum()
        print(f"{coluna}: {preenchidos} registros preenchidos")


if __name__ == "__main__":
    atualizar_features_passagens()