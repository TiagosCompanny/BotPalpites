import os
import re
import json
import time
import math
import statistics
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


TERMO_BUSCA = "Rodovia (5 minutos): quantos carros?"
QTD_HISTORICO = 12
QTD_MINIMA_HISTORICO = 4
INTERVALO_SEGUNDOS = 60

BASE_DIR = Path(__file__).resolve().parent
PASTA_LOGS = BASE_DIR / "LogPrevisoesRodovias"
ARQUIVO_LOG = PASTA_LOGS / "log_previsoes_rodovias.xlsx"


def criar_sessao():
    session = requests.Session()

    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
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
    headers = {"User-Agent": "Mozilla/5.0"}

    response = session.get(montar_url_mercado(market_id), headers=headers, timeout=60)

    response.raise_for_status()
    return response.text


def normalizar_html(html):
    return html.replace('\\"', '"').replace("\\/", "/").replace("&quot;", '"')


def formatar_segundos(segundos):
    segundos = int(segundos)
    horas = segundos // 3600
    minutos = (segundos % 3600) // 60
    segs = segundos % 60
    return f"{horas:02d}:{minutos:02d}:{segs:02d}"


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
        r'"metadata"\s*:\s*\{.*?"valueFinal"\s*:\s*(\d+)',
        r'"valueFinal"\s*:\s*(\d+)',
    ]

    return extrair_primeiro_inteiro(html, padroes)


def extrair_graph_data_points_count(html):
    padroes = [r'"graphDataPointsCount"\s*:\s*(\d+)']

    return extrair_primeiro_inteiro(html, padroes)


def obter_lista_carros_por_market(market_id):
    try:
        html = obter_html_mercado(market_id)
    except Exception as e:
        print(f"Erro ao buscar HTML do mercado {market_id}: {e}")
        return []

    html = normalizar_html(html)

    match = re.search(r'"graphData"\s*:\s*(\[.*?\])', html, re.DOTALL)

    if not match:
        return []

    try:
        dados = json.loads(match.group(1))
    except Exception as e:
        print(f"Erro ao converter graphData do mercado {market_id}: {e}")
        return []

    resultado = []

    for i, item in enumerate(dados, start=1):
        timestamp = (
            item.get("t") or item.get("time") or item.get("timestamp") or item.get("x")
        )

        valor = (
            item.get("v")
            or item.get("value")
            or item.get("y")
            or item.get("count")
            or item.get("carros")
        )

        if valor is None:
            valor = 1

        if timestamp is not None:
            try:
                timestamp_int = int(timestamp)
                hora = datetime.fromtimestamp(timestamp_int).strftime("%H:%M:%S")
            except Exception:
                timestamp_int = None
                hora = str(timestamp)
        else:
            timestamp_int = None
            hora = None

        if not resultado:
            segundos_desde_inicio = 0
            timespan = "00:00:00"
        else:
            timestamp_inicio = resultado[0].get("timestamp")

            if timestamp_int is not None and timestamp_inicio is not None:
                segundos_desde_inicio = timestamp_int - timestamp_inicio
                timespan = formatar_segundos(segundos_desde_inicio)
            else:
                segundos_desde_inicio = None
                timespan = None

        resultado.append(
            {
                "ordem": i,
                "timestamp": timestamp_int,
                "hora": hora,
                "timespan": timespan,
                "segundos_desde_inicio": segundos_desde_inicio,
                "carros": int(valor),
            }
        )

    return resultado


def obter_quantidade_carros_mercado(market_id):
    html = obter_html_mercado(market_id)

    value_final = extrair_value_final(html)

    if value_final is not None:
        return value_final, "valueFinal"

    graph_points_count = extrair_graph_data_points_count(html)

    if graph_points_count is not None:
        return graph_points_count, "graphDataPointsCount"

    dados = obter_lista_carros_por_market(market_id)

    if dados:
        return sum(int(d["carros"]) for d in dados), "graphData"

    return None, "nao_encontrado"


def obter_mercados(status, limit=100, page=1, order_direction="DESC"):
    url = "https://app.palpita.io/api/v1/markets"

    params = {
        "page": page,
        "limit": limit,
        "status": status,
        "search": TERMO_BUSCA,
        "orderBy": "closesAt",
        "orderDirection": order_direction,
    }

    headers = {"Accept": "application/json", "User-Agent": "Mozilla/5.0"}

    response = session.get(url, params=params, headers=headers, timeout=30)
    response.raise_for_status()

    return response.json().get("data", {}).get("items", [])


def obter_mercados_abertos():
    return obter_mercados(status="OPEN", limit=100, order_direction="ASC")


def obter_mercados_resolvidos_historico(limit=QTD_HISTORICO, tag=None):
    mercados = obter_mercados(status="RESOLVED", limit=100, order_direction="DESC")

    if tag:
        mesma_tag = [m for m in mercados if m.get("tag") == tag]

        if len(mesma_tag) >= QTD_MINIMA_HISTORICO:
            mercados = mesma_tag

    return mercados[:limit]


def extrair_meta(mercado):
    for s in mercado.get("selections", []):
        label = s.get("label", "")
        match = re.search(r"\d+", label)

        if match:
            return int(match.group())

    return None


def extrair_rodovia_descricao(description):
    if not description:
        return None

    primeira_linha = description.splitlines()[0].strip()
    return primeira_linha.lstrip("•").strip()


def calcular_media_ponderada_recencia(valores):
    pesos = list(range(len(valores), 0, -1))
    soma_pesos = sum(pesos)

    if soma_pesos == 0:
        return 0

    return sum(v * p for v, p in zip(valores, pesos)) / soma_pesos


def calcular_tendencia_linear(valores):
    n = len(valores)

    if n < 2:
        return 0

    x = list(range(n))
    y = valores

    media_x = statistics.mean(x)
    media_y = statistics.mean(y)

    numerador = sum((x[i] - media_x) * (y[i] - media_y) for i in range(n))
    denominador = sum((x[i] - media_x) ** 2 for i in range(n))

    if denominador == 0:
        return 0

    return numerador / denominador


def calcular_estatisticas(totais):
    media_simples = statistics.mean(totais)
    media_ponderada = calcular_media_ponderada_recencia(totais)
    mediana = statistics.median(totais)

    if len(totais) >= 2:
        desvio = statistics.stdev(totais)
    else:
        desvio = 0

    minimo = min(totais)
    maximo = max(totais)
    amplitude = maximo - minimo
    tendencia = calcular_tendencia_linear(list(reversed(totais)))

    coef_variacao = desvio / media_simples if media_simples > 0 else 0

    previsao_base = media_ponderada * 0.50 + mediana * 0.30 + media_simples * 0.20

    ajuste_tendencia = tendencia * 0.35
    previsao_final = previsao_base + ajuste_tendencia

    limite_inferior = previsao_final - desvio
    limite_superior = previsao_final + desvio

    return {
        "media_simples": round(media_simples, 2),
        "media_ponderada_recencia": round(media_ponderada, 2),
        "mediana": round(mediana, 2),
        "desvio_padrao": round(desvio, 2),
        "coeficiente_variacao": round(coef_variacao, 4),
        "minimo": minimo,
        "maximo": maximo,
        "amplitude": amplitude,
        "tendencia_por_mercado": round(tendencia, 4),
        "ajuste_tendencia": round(ajuste_tendencia, 2),
        "previsao_final": round(previsao_final, 2),
        "limite_inferior_estimado": round(limite_inferior, 2),
        "limite_superior_estimado": round(limite_superior, 2),
    }


def calcular_metricas_processamentos(processamentos):
    if not processamentos:
        return {
            "qtd_processamentos": 0,
            "duracao_observada_segundos": 0,
            "intervalo_medio_entre_carros": None,
            "taxa_carros_por_minuto": None,
        }

    segundos = [
        p.get("segundos_desde_inicio")
        for p in processamentos
        if p.get("segundos_desde_inicio") is not None
    ]

    if not segundos:
        return {
            "qtd_processamentos": len(processamentos),
            "duracao_observada_segundos": 0,
            "intervalo_medio_entre_carros": None,
            "taxa_carros_por_minuto": None,
        }

    duracao = max(segundos) - min(segundos)

    if len(processamentos) > 1 and duracao > 0:
        intervalo_medio = duracao / (len(processamentos) - 1)
        taxa_minuto = len(processamentos) / (duracao / 60)
    else:
        intervalo_medio = None
        taxa_minuto = None

    return {
        "qtd_processamentos": len(processamentos),
        "duracao_observada_segundos": duracao,
        "intervalo_medio_entre_carros": (
            round(intervalo_medio, 2) if intervalo_medio else None
        ),
        "taxa_carros_por_minuto": round(taxa_minuto, 2) if taxa_minuto else None,
    }


def calcular_confianca_avancada(previsao, meta, estatisticas, qtd_historico):
    distancia = abs(previsao - meta)
    desvio = estatisticas["desvio_padrao"]
    coef_variacao = estatisticas["coeficiente_variacao"]

    score = 0.50

    if distancia >= 8:
        score += 0.22
    elif distancia >= 5:
        score += 0.16
    elif distancia >= 3:
        score += 0.10
    elif distancia >= 2:
        score += 0.05
    else:
        score -= 0.08

    if qtd_historico >= 12:
        score += 0.08
    elif qtd_historico >= 8:
        score += 0.05
    elif qtd_historico >= 5:
        score += 0.03
    else:
        score -= 0.05

    if desvio <= 2:
        score += 0.10
    elif desvio <= 4:
        score += 0.06
    elif desvio <= 6:
        score += 0.02
    else:
        score -= 0.08

    if coef_variacao <= 0.10:
        score += 0.06
    elif coef_variacao <= 0.20:
        score += 0.03
    else:
        score -= 0.05

    return round(max(0.35, min(score, 0.92)), 2)


def interpretar_risco(confianca, previsao, meta):
    distancia = abs(previsao - meta)

    if confianca >= 0.78 and distancia >= 4:
        return "FORTE"

    if confianca >= 0.65 and distancia >= 2:
        return "MODERADO"

    return "FRACO"


def prever_mercado_aberto_avancado_por_mercado(mercado_aberto):
    meta = extrair_meta(mercado_aberto)

    if meta is None:
        return {
            "erro": "Não foi possível identificar a meta do mercado aberto",
            "market_id_aberto": mercado_aberto.get("id"),
        }

    tag_aberta = mercado_aberto.get("tag")
    mercados = obter_mercados_resolvidos_historico(limit=QTD_HISTORICO, tag=tag_aberta)

    if len(mercados) < QTD_MINIMA_HISTORICO:
        return {
            "erro": "Histórico insuficiente para previsão avançada",
            "market_id_aberto": mercado_aberto.get("id"),
            "qtd_encontrada": len(mercados),
        }

    base = []
    totais = []

    for mercado in mercados:
        market_id = mercado.get("id")

        try:
            processamentos = obter_lista_carros_por_market(market_id)
            qtd, origem = obter_quantidade_carros_mercado(market_id)

            total = (
                qtd
                if qtd is not None
                else sum(int(d["carros"]) for d in processamentos)
            )

            if total is None or total <= 0:
                continue

            metricas_proc = calcular_metricas_processamentos(processamentos)
            meta_mercado = extrair_meta(mercado)

            if meta_mercado is not None and total >= meta_mercado + 1:
                resultado_real = "Mais"
            else:
                resultado_real = "Até"

            totais.append(total)

            base.append(
                {
                    "market_id": market_id,
                    "tag": mercado.get("tag"),
                    "status": mercado.get("status"),
                    "total_carros": total,
                    "origem_total": origem,
                    "meta_mercado": meta_mercado,
                    "resultado_real": resultado_real,
                    "qtd_processamentos": metricas_proc["qtd_processamentos"],
                    "duracao_observada_segundos": metricas_proc[
                        "duracao_observada_segundos"
                    ],
                    "intervalo_medio_entre_carros": metricas_proc[
                        "intervalo_medio_entre_carros"
                    ],
                    "taxa_carros_por_minuto": metricas_proc["taxa_carros_por_minuto"],
                    "opensAt": mercado.get("opensAt"),
                    "closesAt": mercado.get("closesAt"),
                    "resultSelectionId": mercado.get("resultSelectionId"),
                }
            )

            time.sleep(0.25)

        except Exception as e:
            print(f"Erro ao processar mercado {market_id}: {e}")

    if len(totais) < QTD_MINIMA_HISTORICO:
        return {
            "erro": "Poucos mercados válidos com total de carros",
            "market_id_aberto": mercado_aberto.get("id"),
            "qtd_validos": len(totais),
        }

    estatisticas = calcular_estatisticas(totais)
    previsao_final = estatisticas["previsao_final"]

    if previsao_final >= meta + 1:
        resultado_previsto = f"Mais de {meta}"
        direcao = "MAIS"
    else:
        resultado_previsto = f"Até {meta}"
        direcao = "ATE"

    confianca = calcular_confianca_avancada(
        previsao=previsao_final,
        meta=meta,
        estatisticas=estatisticas,
        qtd_historico=len(totais),
    )

    forca_entrada = interpretar_risco(
        confianca=confianca, previsao=previsao_final, meta=meta
    )

    return {
        "market_id_aberto": mercado_aberto.get("id"),
        "rodovia": extrair_rodovia_descricao(mercado_aberto.get("description", "")),
        "tag": tag_aberta,
        "meta": meta,
        "resultado_previsto": resultado_previsto,
        "direcao": direcao,
        "media_carros_prevista": previsao_final,
        "confianca": confianca,
        "forca_entrada": forca_entrada,
        "estatisticas": estatisticas,
        "qtd_mercados_usados": len(totais),
        "totais_usados": totais,
        "base_historica": base,
    }


def garantir_pasta_logs():
    PASTA_LOGS.mkdir(parents=True, exist_ok=True)


def salvar_log_previsao(previsao):
    garantir_pasta_logs()

    linha = {
        "data_hora_log": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "market_id_aberto": previsao.get("market_id_aberto"),
        "rodovia": previsao.get("rodovia"),
        "tag": previsao.get("tag"),
        "meta": previsao.get("meta"),
        "resultado_previsto": previsao.get("resultado_previsto"),
        "direcao": previsao.get("direcao"),
        "carros_previstos": previsao.get("media_carros_prevista"),
        "confianca": previsao.get("confianca"),
        "forca_entrada": previsao.get("forca_entrada"),
        "qtd_mercados_usados": previsao.get("qtd_mercados_usados"),
        "totais_usados": json.dumps(
            previsao.get("totais_usados", []), ensure_ascii=False
        ),
    }

    estatisticas = previsao.get("estatisticas", {})

    for chave, valor in estatisticas.items():
        linha[f"estat_{chave}"] = valor

    base_historica = previsao.get("base_historica", [])

    linha["base_historica_json"] = json.dumps(base_historica, ensure_ascii=False)

    df_novo = pd.DataFrame([linha])

    if ARQUIVO_LOG.exists():
        df_existente = pd.read_excel(ARQUIVO_LOG)
        df_final = pd.concat([df_existente, df_novo], ignore_index=True)
    else:
        df_final = df_novo

    df_final.to_excel(ARQUIVO_LOG, index=False)


def salvar_log_erro(market_id, erro):
    garantir_pasta_logs()

    linha = {
        "data_hora_log": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "market_id_aberto": market_id,
        "erro": erro,
    }

    df_novo = pd.DataFrame([linha])
    arquivo_erro = PASTA_LOGS / "log_erros_previsoes_rodovias.xlsx"

    if arquivo_erro.exists():
        df_existente = pd.read_excel(arquivo_erro)
        df_final = pd.concat([df_existente, df_novo], ignore_index=True)
    else:
        df_final = df_novo

    df_final.to_excel(arquivo_erro, index=False)


def rodar_observador_todos_mercados():
    garantir_pasta_logs()

    print("Observador iniciado.")
    print(f"Log de previsões: {ARQUIVO_LOG}")

    mercados_ja_processados = set()

    while True:
        try:
            mercados_abertos = obter_mercados_abertos()

            if not mercados_abertos:
                time.sleep(10)
                continue

            for mercado in mercados_abertos:
                market_id = mercado.get("id")

                if market_id in mercados_ja_processados:
                    continue

                previsao = prever_mercado_aberto_avancado_por_mercado(mercado)

                if previsao.get("erro"):
                    erro = previsao.get("erro")
                    print(f"Mercado {market_id}: {erro}")
                    salvar_log_erro(market_id, erro)
                    mercados_ja_processados.add(market_id)
                    continue

                salvar_log_previsao(previsao)
                mercados_ja_processados.add(market_id)

                print(
                    f"NOVO MERCADO PROCESSADO | "
                    f"Mercado {market_id} | "
                    f"{previsao.get('rodovia')} | "
                    f"Meta: {previsao.get('meta')} | "
                    f"Previsto: {previsao.get('resultado_previsto')} | "
                    f"Carros: {previsao.get('media_carros_prevista')} | "
                    f"Confiança: {previsao.get('confianca')} | "
                    f"Força: {previsao.get('forca_entrada')}"
                )

            time.sleep(10)

        except KeyboardInterrupt:
            print("Observador encerrado manualmente.")
            break

        except Exception as e:
            print(f"Erro geral no observador: {e}")
            time.sleep(10)


if __name__ == "__main__":
    rodar_observador_todos_mercados()
