import os
import re
import time
import requests
import pandas as pd
from pathlib import Path
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_DIR = Path(_file_).resolve().parent
ARQUIVO_BASE = BASE_DIR.parent / "DadosRodovias" / "dados_todas_rodovias.xlsx"

COLUNA_ID_MERCADO = "id"
COLUNA_QTD_CARROS = "QuantidadeCarros"


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


def garantir_coluna_quantidade_carros(df):
    if COLUNA_QTD_CARROS not in df.columns:
        df[COLUNA_QTD_CARROS] = pd.NA
    return df


def montar_url_mercado(market_id):
    market_id = int(market_id)
    return f"https://app.palpita.io/live/{market_id}-market/rodovia-5-minutos-qu-{market_id}"


def obter_html_mercado(market_id):
    url = montar_url_mercado(market_id)

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = session.get(url, headers=headers, timeout=60)
    response.raise_for_status()
    return response.text


def normalizar_html(html):
    return (
        html
        .replace('\\"', '"')
        .replace("\\/", "/")
        .replace("&quot;", '"')
    )


def extrair_primeiro_inteiro(html, padroes):
    html_norm = normalizar_html(html)

    for padrao in padroes:
        match = re.search(padrao, html_norm, re.DOTALL)
        if match:
            try:
                return int(match.group(1))
            except Exception:
                pass

    return None


def extrair_value_final(html):
    padroes = [
        r'"metadata"\s*:\s*\{.?"valueFinal"\s:\s*(\d+)',
        r'"valueFinal"\s*:\s*(\d+)'
    ]
    return extrair_primeiro_inteiro(html, padroes)


def extrair_graph_data_points_count(html):
    padroes = [
        r'"graphDataPointsCount"\s*:\s*(\d+)'
    ]
    return extrair_primeiro_inteiro(html, padroes)


def obter_quantidade_carros_mercado(market_id):
    html = obter_html_mercado(market_id)

    value_final = extrair_value_final(html)
    if value_final is not None:
        return value_final, "valueFinal"

    graph_points_count = extrair_graph_data_points_count(html)
    if graph_points_count is not None:
        return graph_points_count, "graphDataPointsCount"

    return None, "nao_encontrado"


def formatar_segundos(segundos):
    segundos = int(segundos)
    horas = segundos // 3600
    minutos = (segundos % 3600) // 60
    segs = segundos % 60
    return f"{horas:02d}:{minutos:02d}:{segs:02d}"


def atualizar_quantidade_carros(caminho_arquivo):
    if not os.path.exists(caminho_arquivo):
        print(f"Arquivo não encontrado: {caminho_arquivo}")
        return

    df = pd.read_excel(caminho_arquivo)
    print(f"Linhas carregadas: {len(df)}")

    if df.empty:
        print("Planilha vazia.")
        return

    if COLUNA_ID_MERCADO not in df.columns:
        print(f"Coluna não encontrada: {COLUNA_ID_MERCADO}")
        return

    df = garantir_coluna_quantidade_carros(df)

    indices_pendentes = [
        idx for idx in df.index
        if pd.notna(df.at[idx, COLUNA_ID_MERCADO]) and pd.isna(df.at[idx, COLUNA_QTD_CARROS])
    ]

    total_pendentes = len(indices_pendentes)

    if total_pendentes == 0:
        print("Nenhuma linha pendente para atualizar.")
        return

    total_atualizadas = 0
    total_nao_encontrado = 0
    total_erros = 0
    inicio = time.time()

    for posicao, idx in enumerate(indices_pendentes, start=1):
        market_id = df.at[idx, COLUNA_ID_MERCADO]

        try:
            qtd, origem = obter_quantidade_carros_mercado(market_id)

            if qtd is not None:
                df.at[idx, COLUNA_QTD_CARROS] = qtd
                total_atualizadas += 1
                mensagem_resultado = f"Mercado {market_id} -> {qtd} carros [{origem}]"
            else:
                total_nao_encontrado += 1
                mensagem_resultado = f"Mercado {market_id} -> valor não encontrado"

            decorrido = time.time() - inicio
            media = decorrido / posicao
            restante = media * (total_pendentes - posicao)
            percentual = (posicao / total_pendentes) * 100

            print(
                f"[{posicao}/{total_pendentes}] "
                f"{percentual:.1f}% | "
                f"{mensagem_resultado} | "
                f"Decorrido: {formatar_segundos(decorrido)} | "
                f"Estimado restante: {formatar_segundos(restante)}"
            )

            if posicao % 50 == 0:
                df.to_excel(caminho_arquivo, index=False)
                print(f"Progresso salvo em disco na linha {posicao}")

            time.sleep(0.5)

        except Exception as e:
            total_erros += 1

            decorrido = time.time() - inicio
            media = decorrido / posicao
            restante = media * (total_pendentes - posicao)
            percentual = (posicao / total_pendentes) * 100

            print(
                f"[{posicao}/{total_pendentes}] "
                f"{percentual:.1f}% | "
                f"Erro no mercado {market_id}: {e} | "
                f"Decorrido: {formatar_segundos(decorrido)} | "
                f"Estimado restante: {formatar_segundos(restante)}"
            )

    if "fechamento" in df.columns:
        df["fechamento"] = pd.to_datetime(df["fechamento"], errors="coerce")
        df = df.sort_values(by="fechamento", ascending=False)

    df.to_excel(caminho_arquivo, index=False)

    tempo_total = time.time() - inicio

    print()
    print(f"Linhas atualizadas em {COLUNA_QTD_CARROS}: {total_atualizadas}")
    print(f"Linhas sem valor encontrado: {total_nao_encontrado}")
    print(f"Total de erros: {total_erros}")
    print(f"Tempo total: {formatar_segundos(tempo_total)}")
    print(f"Planilha atualizada com sucesso: {caminho_arquivo}")


atualizar_quantidade_carros(ARQUIVO_BASE)