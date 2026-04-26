import os
import re
import json
import time
import requests
import pandas as pd
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from pathlib import Path



BASE_DIR = Path(__file__).resolve().parent
ARQUIVO_BASE = BASE_DIR / "DadosRodovias" / "dados_todas_rodovias.xlsx"
ARQUIVO_SAIDA = BASE_DIR / "DadosRodovias" / "passagens_carros_por_mercado.xlsx"
COLUNA_ID_MERCADO = "id"


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
        .replace("\\u003c", "<")
        .replace("\\u003e", ">")
        .replace("\\u0026", "&")
    )


def extrair_array_json_balanceado(texto, nome_campo):
    texto = normalizar_html(texto)

    pos = texto.find(f'"{nome_campo}"')

    if pos == -1:
        return None

    inicio_array = texto.find("[", pos)

    if inicio_array == -1:
        return None

    nivel = 0
    dentro_string = False
    escape = False

    for i in range(inicio_array, len(texto)):
        char = texto[i]

        if escape:
            escape = False
            continue

        if char == "\\":
            escape = True
            continue

        if char == '"':
            dentro_string = not dentro_string
            continue

        if dentro_string:
            continue

        if char == "[":
            nivel += 1

        elif char == "]":
            nivel -= 1

            if nivel == 0:
                return texto[inicio_array:i + 1]

    return None


def obter_graph_data_mercado(market_id):
    html = obter_html_mercado(market_id)

    texto_array = extrair_array_json_balanceado(
        html,
        "graphData"
    )

    if not texto_array:
        return []

    try:
        return json.loads(texto_array)
    except Exception as e:
        print(f"Erro ao converter graphData do mercado {market_id}: {e}")
        return []


def converter_timestamp_para_data_hora(timestamp):
    if timestamp is None:
        return None

    try:
        return datetime.fromtimestamp(int(timestamp)).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def formatar_segundos(segundos):
    segundos = int(segundos)
    horas = segundos // 3600
    minutos = (segundos % 3600) // 60
    segs = segundos % 60
    return f"{horas:02d}:{minutos:02d}:{segs:02d}"


def gerar_base_passagens():
    if not os.path.exists(ARQUIVO_BASE):
        print(f"Arquivo base não encontrado: {ARQUIVO_BASE}")
        return

    df_mercados = pd.read_excel(ARQUIVO_BASE)

    if COLUNA_ID_MERCADO not in df_mercados.columns:
        print(f"Coluna '{COLUNA_ID_MERCADO}' não encontrada na planilha.")
        return

    ids_mercados = (
        df_mercados[COLUNA_ID_MERCADO]
        .dropna()
        .astype(int)
        .drop_duplicates()
        .tolist()
    )

    registros = []
    total = len(ids_mercados)
    inicio = time.time()

    for posicao, mercado_id in enumerate(ids_mercados, start=1):
        try:
            graph_data = obter_graph_data_mercado(mercado_id)

            for item in graph_data:
                registros.append({
                    "MercadoId": mercado_id,
                    "Value": item.get("value"),
                    "DataHora": converter_timestamp_para_data_hora(item.get("timestamp"))
                })

            percentual = (posicao / total) * 100
            decorrido = time.time() - inicio
            media_por_mercado = decorrido / posicao
            restante = media_por_mercado * (total - posicao)

            print(
                f"[{posicao}/{total}] "
                f"{percentual:.2f}% | "
                f"Mercado {mercado_id} -> {len(graph_data)} passagens | "
                f"Total acumulado: {len(registros)} | "
                f"Decorrido: {formatar_segundos(decorrido)} | "
                f"Restante estimado: {formatar_segundos(restante)}"
            )

            time.sleep(0.5)

        except Exception as e:
            percentual = (posicao / total) * 100
            decorrido = time.time() - inicio
            media_por_mercado = decorrido / posicao
            restante = media_por_mercado * (total - posicao)

            print(
                f"[{posicao}/{total}] "
                f"{percentual:.2f}% | "
                f"Erro no mercado {mercado_id}: {e} | "
                f"Decorrido: {formatar_segundos(decorrido)} | "
                f"Restante estimado: {formatar_segundos(restante)}"
            )

    df_saida = pd.DataFrame(registros)

    if not df_saida.empty:
        df_saida = df_saida.sort_values(
            by=["MercadoId", "DataHora"],
            ascending=[True, True]
        )

    df_saida.to_excel(ARQUIVO_SAIDA, index=False)

    tempo_total = time.time() - inicio

    print()
    print(f"Total de mercados lidos: {total}")
    print(f"Total de passagens geradas: {len(df_saida)}")
    print(f"Tempo total: {formatar_segundos(tempo_total)}")
    print("Arquivo gerado com sucesso:")
    print(ARQUIVO_SAIDA)


gerar_base_passagens()