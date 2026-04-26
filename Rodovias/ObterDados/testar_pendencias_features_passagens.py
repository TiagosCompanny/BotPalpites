import os
import pandas as pd
from pathlib import Path

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


def normalizar_id(valor):
    if pd.isna(valor):
        return None

    texto = str(valor).strip()

    if texto == "":
        return None

    try:
        numero = float(texto)
        if numero.is_integer():
            return str(int(numero))
    except Exception:
        pass

    return texto


def carregar_bases():
    if not os.path.exists(ARQUIVO_BASE):
        raise FileNotFoundError(f"Base principal não encontrada: {ARQUIVO_BASE}")

    if not os.path.exists(ARQUIVO_PASSAGENS):
        raise FileNotFoundError(f"Base de passagens não encontrada: {ARQUIVO_PASSAGENS}")

    df_base = pd.read_excel(ARQUIVO_BASE)
    df_passagens = pd.read_excel(ARQUIVO_PASSAGENS)

    df_base["id"] = df_base["id"].apply(normalizar_id)
    df_base["abertura"] = pd.to_datetime(df_base["abertura"], errors="coerce")
    df_base["fechamento"] = pd.to_datetime(df_base["fechamento"], errors="coerce")

    df_passagens["MercadoId"] = df_passagens["MercadoId"].apply(normalizar_id)
    df_passagens["DataHora"] = pd.to_datetime(df_passagens["DataHora"], errors="coerce")

    return df_base, df_passagens


def contar_passagens(df_passagens, mercado_id):
    df_mercado = df_passagens[df_passagens["MercadoId"] == str(mercado_id)].copy()

    return {
        "qtd_passagens": len(df_mercado),
        "primeira_passagem": df_mercado["DataHora"].min() if not df_mercado.empty else None,
        "ultima_passagem": df_mercado["DataHora"].max() if not df_mercado.empty else None,
    }


def testar_pendencias():
    df_base, df_passagens = carregar_bases()

    for coluna in COLUNAS_FEATURES_PASSAGENS:
        if coluna not in df_base.columns:
            df_base[coluna] = pd.NA

    mascara_pendente = df_base[COLUNAS_FEATURES_PASSAGENS].isna().any(axis=1)

    df_pendentes = df_base[mascara_pendente].copy()

    print("")
    print("===== TESTE DE PENDÊNCIAS FEATURES PASSAGENS =====")
    print(f"Arquivo base: {ARQUIVO_BASE}")
    print(f"Arquivo passagens: {ARQUIVO_PASSAGENS}")
    print(f"Total de linhas na base: {len(df_base)}")
    print(f"Total de linhas pendentes: {len(df_pendentes)}")
    print("")

    if df_pendentes.empty:
        print("Nenhuma pendência encontrada.")
        return

    df_base_ordenada = df_base.sort_values(
        by=["rodovia_identificacao", "abertura"],
        ascending=[True, True]
    ).reset_index(drop=True)

    for _, pendente in df_pendentes.iterrows():
        mercado_id = str(pendente["id"])
        rodovia = str(pendente["rodovia_identificacao"])
        abertura = pendente["abertura"]
        fechamento = pendente["fechamento"] if "fechamento" in pendente else None

        print("")
        print("--------------------------------------------------")
        print(f"Mercado pendente: {mercado_id}")
        print(f"Rodovia: {rodovia}")
        print(f"Abertura: {abertura}")
        print(f"Fechamento: {fechamento}")

        colunas_vazias = [
            coluna for coluna in COLUNAS_FEATURES_PASSAGENS
            if pd.isna(pendente.get(coluna))
        ]

        print(f"Campos vazios: {colunas_vazias}")

        info_atual = contar_passagens(df_passagens, mercado_id)

        print("")
        print("Passagens do mercado atual:")
        print(f"Quantidade: {info_atual['qtd_passagens']}")
        print(f"Primeira passagem: {info_atual['primeira_passagem']}")
        print(f"Última passagem: {info_atual['ultima_passagem']}")

        df_mesma_rodovia = df_base_ordenada[
            df_base_ordenada["rodovia_identificacao"].astype(str) == rodovia
        ].copy()

        indices = df_mesma_rodovia.index[
            df_mesma_rodovia["id"].astype(str) == mercado_id
        ].tolist()

        if not indices:
            print("")
            print("Motivo provável: mercado não encontrado na base ordenada da rodovia.")
            continue

        idx_global = indices[0]
        posicoes = df_mesma_rodovia.index.tolist()
        posicao_na_rodovia = posicoes.index(idx_global)

        print("")
        print(f"Posição desse mercado dentro da rodovia: {posicao_na_rodovia + 1} de {len(df_mesma_rodovia)}")

        if posicao_na_rodovia == 0:
            print("Motivo provável: este é o primeiro mercado da rodovia. Não existe mercado anterior para calcular lag/roll.")
            continue

        idx_anterior = posicoes[posicao_na_rodovia - 1]
        mercado_anterior = df_mesma_rodovia.loc[idx_anterior]

        mercado_anterior_id = str(mercado_anterior["id"])
        abertura_anterior = mercado_anterior["abertura"]
        fechamento_anterior = mercado_anterior["fechamento"] if "fechamento" in mercado_anterior else None

        print("")
        print("Mercado anterior encontrado:")
        print(f"ID anterior: {mercado_anterior_id}")
        print(f"Abertura anterior: {abertura_anterior}")
        print(f"Fechamento anterior: {fechamento_anterior}")

        info_anterior = contar_passagens(df_passagens, mercado_anterior_id)

        print("")
        print("Passagens do mercado anterior:")
        print(f"Quantidade: {info_anterior['qtd_passagens']}")
        print(f"Primeira passagem: {info_anterior['primeira_passagem']}")
        print(f"Última passagem: {info_anterior['ultima_passagem']}")

        if info_anterior["qtd_passagens"] == 0:
            print("")
            print("Motivo provável: o mercado anterior não tem passagens registradas na passagens_carros_por_mercado.xlsx.")
        else:
            print("")
            print("O mercado anterior tem passagens. Se ainda ficou pendente, pode ser problema de alinhamento por ID ou ordenação.")

    print("")
    print("===== FIM DO TESTE =====")


if __name__ == "__main__":
    testar_pendencias()