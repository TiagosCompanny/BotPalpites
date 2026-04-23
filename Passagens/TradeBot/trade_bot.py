import os
import json
import base64
from datetime import datetime
from pathlib import Path
import time
import schedule
import joblib
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime, timezone, timedelta


CAMINHO_MODELO = r"C:\Users\Tiago Carvalho\Desktop\Projeto Palpites\BotPalpites\Passagens\modelos_passagens\rua_passagens_rf_regimes"

BUNDLE = joblib.load(os.path.join(CAMINHO_MODELO, "modelo.joblib"))

CONFIG_ENTRADA_PASSAGENS = {
    "Rua (4m 40s): quantas passagens?": {
        "threshold_confianca": 0.70,
        "odd_minima_base": 2.30,
        "degrau_odds": [0.25, 0.10, 0.00, -0.10]
    }
}

def carregar_variaveis_ambiente():
    raiz_projeto = Path(__file__).resolve().parents[2]
    caminho_env = raiz_projeto / ".env"

    if not caminho_env.exists():
        return

    for linha in caminho_env.read_text(encoding="utf-8").splitlines():
        linha = linha.strip()
        if not linha or linha.startswith("#") or "=" not in linha:
            continue

        chave, valor = linha.split("=", 1)
        chave = chave.strip()
        valor = valor.strip().strip('"').strip("'")

        if chave:
            os.environ.setdefault(chave, valor)


carregar_variaveis_ambiente()

api_key = os.getenv("PALPITA_API_KEY")
api_secret = os.getenv("PALPITA_API_SECRET")

if not api_key or not api_secret:
    raise RuntimeError(
        "Defina PALPITA_API_KEY e PALPITA_API_SECRET no ambiente ou no arquivo .env na raiz do projeto."
    )

market_id_em_andamento = None
order_id_em_andamento = None

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

def obter_bearer_token(api_key, api_secret):
    return base64.b64encode(f"{api_key}:{api_secret}".encode()).decode()

def obter_headers_autenticados():
    token = obter_bearer_token(api_key, api_secret)
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "Content-Type": "application/json"
    }

def obter_saldo():
    headers = obter_headers_autenticados()
    url = "https://app.palpita.io/api/v1/balance"

    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        saldo = response.json()["data"][0]["amount"]
        return saldo
    except requests.HTTPError:
        print("Erro HTTP:", response.status_code, response.text)
    except Exception as e:
        print("Erro:", str(e))

    return None

def obter_mercado_passagens_aberto():
    url = "https://app.palpita.io/api/v1/markets"
    params = {
        "page": 1,
        "limit": 100,
        "search": "Rua (4m 40s): quantas passagens?",
        "orderBy": "closesAt",
        "orderDirection": "ASC"
    }
    headers = {
        "Accept": "application/json"
    }

    response = requests.get(url, headers=headers, params=params, timeout=30)
    response.raise_for_status()

    resposta = response.json()
    mercados = resposta.get("data", {}).get("items", [])
    mercado_aberto = next((m for m in mercados if m.get("status") == "OPEN"), None)

    return mercado_aberto

def pode_enviar_ordem(mercado):
    if not mercado:
        return False

    if mercado.get("status") != "OPEN":
        return False

    closes_at = mercado.get("closesAt")
    if not closes_at:
        return False

    agora = datetime.now().astimezone()
    limite = datetime.fromisoformat(closes_at)

    return agora < limite

def pode_apostar_no_primeiro_minuto(mercado):
    if not mercado:
        return False

    opens_at = mercado.get("opensAt")
    if not opens_at:
        return False

    agora = datetime.now().astimezone()
    abertura = datetime.fromisoformat(opens_at)

    segundos_passados = (agora - abertura).total_seconds()

    return 0 <= segundos_passados <= 60

def extrair_meta_referencia_do_mercado(mercado):
    value_needed = mercado.get("valueNeeded")
    if value_needed is not None:
        return float(value_needed)

    for selection in mercado.get("selections", []):
        label = str(selection.get("label", "")).strip()

        if label.startswith("Mais de "):
            return float(label.replace("Mais de ", "").strip())

        if label.startswith("Até "):
            return float(label.replace("Até ", "").strip())

    return None

def obter_clima_toquio():
    try:
        url = "https://api.open-meteo.com/v1/forecast"

        agora_tokyo = datetime.now(timezone.utc).astimezone(
            timezone(timedelta(hours=9))
        )

        hora_dt = pd.to_datetime(agora_tokyo.strftime("%Y-%m-%dT%H:00"))

        params = {
            "latitude": 35.6895,
            "longitude": 139.6917,
            "hourly": "temperature_2m,relative_humidity_2m,precipitation,cloud_cover",
            "timezone": "Asia/Tokyo"
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()
        hourly = data.get("hourly", {})

        df = pd.DataFrame({
            "time": hourly.get("time", []),
            "temperatura_2m": hourly.get("temperature_2m", []),
            "umidade_relativa": hourly.get("relative_humidity_2m", []),
            "chuva_mm": hourly.get("precipitation", []),
            "cobertura_nuvens": hourly.get("cloud_cover", []),
        })

        if df.empty:
            return None

        df["time"] = pd.to_datetime(df["time"], errors="coerce")

        # 🔥 pega o horário MAIS PRÓXIMO (robusto)
        df["diff"] = (df["time"] - hora_dt).abs()
        linha = df.sort_values("diff").iloc[0]

        return {
            "temperatura_2m": linha["temperatura_2m"],
            "umidade_relativa": linha["umidade_relativa"],
            "chuva_mm": linha["chuva_mm"],
            "cobertura_nuvens": linha["cobertura_nuvens"],
            "estava_chovendo": int(linha["chuva_mm"] > 0) if pd.notna(linha["chuva_mm"]) else 0
        }

    except Exception as e:
        print("Erro ao obter clima:", e)
        return None
    
    
def prever_aposta(mercado, threshold_confianca):
    meta_referencia = extrair_meta_referencia_do_mercado(mercado)
    agora = datetime.now()

    # Carregar bundle
    bundle = BUNDLE
    modelo_pipeline = bundle["modelo"]
    features_treino = bundle["features"]

    # =========================
    # BASE INICIAL (mínima)
    # =========================
    hora_decimal = agora.hour + (agora.minute / 60.0)

    dados = {
        "hora": agora.hour,
        "minuto": agora.minute,
        "hora_decimal": hora_decimal,
        "dia_semana": agora.weekday(),
        "mes": agora.month,
        "fim_semana": 1 if agora.weekday() in [5, 6] else 0,
        "hora_sin": np.sin(2 * np.pi * hora_decimal / 24),
        "hora_cos": np.cos(2 * np.pi * hora_decimal / 24),
        "dia_semana_sin": np.sin(2 * np.pi * agora.weekday() / 7),
        "dia_semana_cos": np.cos(2 * np.pi * agora.weekday() / 7),
        "mes_sin": np.sin(2 * np.pi * (agora.month - 1) / 12),
        "mes_cos": np.cos(2 * np.pi * (agora.month - 1) / 12),

        # flags
        "eh_sexta": 1 if agora.weekday() == 4 else 0,
        "eh_sabado": 1 if agora.weekday() == 5 else 0,
        "eh_domingo": 1 if agora.weekday() == 6 else 0,
        "eh_dia_util": 1 if agora.weekday() <= 4 else 0,

        # blocos (default)
        "bloco_madrugada": 0,
        "bloco_manha": 0,
        "bloco_almoco": 0,
        "bloco_tarde_pico": 0,
        "bloco_noite": 0,
        "bloco_late_night": 0,
        "bloco_pico_manha": 0,

        # clima default
        "temperatura_2m": np.nan,
        "umidade_relativa": np.nan,
        "cobertura_nuvens": np.nan,
        "chuva_mm": 0,
        "estava_chovendo": 0,
    }

    # =========================
    # BLOCO DO DIA (importante!)
    # =========================
    h = agora.hour

    if 0 <= h < 5:
        dados["bloco_madrugada"] = 1
    elif 5 <= h < 9:
        dados["bloco_manha"] = 1
        dados["bloco_pico_manha"] = 1
    elif 9 <= h < 13:
        dados["bloco_almoco"] = 1
    elif 13 <= h < 18:
        dados["bloco_tarde_pico"] = 1
    elif 18 <= h < 22:
        dados["bloco_noite"] = 1
    else:
        dados["bloco_late_night"] = 1

    # =========================
    # CLIMA
    # =========================
    clima = obter_clima_toquio()
    if clima:
        dados["temperatura_2m"] = clima.get("temperatura_2m", np.nan)
        dados["umidade_relativa"] = clima.get("umidade_relativa", np.nan)
        dados["cobertura_nuvens"] = clima.get("cobertura_nuvens", np.nan)
        dados["chuva_mm"] = clima.get("chuva_mm", 0)
        dados["estava_chovendo"] = clima.get("estava_chovendo", 0)

    df_predict = pd.DataFrame([dados])

    # =========================
    # GARANTIR TODAS FEATURES
    # =========================
    for col in features_treino:
        if col not in df_predict.columns:
            df_predict[col] = 0

    # garantir colunas do cluster
    for col in bundle["colunas_cluster"]:
        if col not in df_predict.columns:
            df_predict[col] = 0
    # =========================
    # CLUSTER (REGIME)
    # =========================
    X_cluster = bundle["imp_cluster"].transform(
        df_predict[bundle["colunas_cluster"]].fillna(0)
    )
    X_cluster_scaled = bundle["scaler_cluster"].transform(X_cluster)
    cluster = bundle["kmeans"].predict(X_cluster_scaled)[0]

    df_predict["cluster_regime"] = cluster

    for i in range(4):
        df_predict[f"cluster_regime_{i}"] = 1 if cluster == i else 0

    df_predict = df_predict[features_treino]
    # =========================
    # PREDIÇÃO
    # =========================
    prob = modelo_pipeline.predict_proba(df_predict)[0][1]

    return {
        "apostar": prob >= threshold_confianca,
        "confianca": prob,
        "previsao": f"Mais de {meta_referencia}" if prob >= 0.5 else f"Até {meta_referencia}",
        "meta_referencia": meta_referencia
    }


def obter_selection_id_da_previsao(mercado, resultado):
    previsao = resultado["previsao"]
    meta = int(float(resultado["meta_referencia"]))

    if previsao.startswith("Mais de"):
        label_esperada = f"Mais de {meta}"
    else:
        label_esperada = f"Até {meta}"

    selection = next(
        (s for s in mercado.get("selections", []) if s.get("label") == label_esperada),
        None
    )

    if selection is None:
        raise ValueError(f"Nenhuma selection encontrada para: {label_esperada}")

    return int(selection["id"])

def criar_ordem_limit_buy(selection_id, valor_total, odd_minima):
    price = round(1 / odd_minima, 2)
    amount = round(valor_total / price, 4)

    if round(amount * price, 4) <= 1:
        amount = round((1.01 / price), 4)

    token = base64.b64encode(f"{api_key}:{api_secret}".encode()).decode()

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "Content-Type": "application/json"
    }

    payload = {
        "selectionId": int(selection_id),
        "side": "BUY",
        "type": "LIMIT",
        "price": f"{price:.2f}",
        "amount": f"{amount:.4f}"
    }

    response = requests.post(
        "https://app.palpita.io/api/v1/orders",
        headers=headers,
        json=payload,
        timeout=30
    )

    print("PAYLOAD:", payload)
    print("STATUS:", response.status_code)
    print("RESPOSTA:", response.text)

    response.raise_for_status()
    return response.json()

def consultar_ordem(order_id):
    token = base64.b64encode(f"{api_key}:{api_secret}".encode()).decode()

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json"
    }

    response = requests.get(
        f"https://app.palpita.io/api/v1/orders/{order_id}",
        headers=headers,
        timeout=30
    )

    print("CONSULTAR ORDEM STATUS:", response.status_code)
    print("CONSULTAR ORDEM RESPOSTA:", response.text)

    response.raise_for_status()
    return response.json()

def cancelar_ordem(order_id):
    token = base64.b64encode(f"{api_key}:{api_secret}".encode()).decode()

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json"
    }

    response = requests.delete(
        f"https://app.palpita.io/api/v1/orders/{order_id}",
        headers=headers,
        timeout=30
    )

    print("CANCELAR ORDEM STATUS:", response.status_code)
    print("CANCELAR ORDEM RESPOSTA:", response.text)

    response.raise_for_status()
    return response.json()

def criar_ordem_market_buy(selection_id, valor_total):
    token = base64.b64encode(f"{api_key}:{api_secret}".encode()).decode()

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "Content-Type": "application/json"
    }

    payload = {
        "selectionId": int(selection_id),
        "side": "BUY",
        "type": "MARKET",
        "total": f"{valor_total:.2f}"
    }

    response = requests.post(
        "https://app.palpita.io/api/v1/orders",
        headers=headers,
        json=payload,
        timeout=30
    )

    print("MARKET BUY PAYLOAD:", payload)
    print("MARKET BUY STATUS:", response.status_code)
    print("MARKET BUY RESPOSTA:", response.text)

    response.raise_for_status()
    return response.json()

def obter_status_e_preenchimento(resposta_ordem):
    data = resposta_ordem.get("data", resposta_ordem)

    status = data.get("status")
    filled_amount = float(data.get("filledAmount", 0) or 0)
    amount = float(data.get("amount", 0) or 0)

    return status, filled_amount, amount

def executar_entrada_escalonada(
    selection_id,
    valor_total,
    odds_tentativa,
    segundos_espera_por_tentativa=3,
    usar_market_no_final=False
):
    historico = []

    for odd in odds_tentativa:
        resposta_ordem = criar_ordem_limit_buy(
            selection_id=selection_id,
            valor_total=valor_total,
            odd_minima=odd
        )

        data_ordem = resposta_ordem.get("data", {})
        order_id = data_ordem.get("orderId") or data_ordem.get("id")

        if not order_id:
            raise ValueError("A API não retornou orderId na criação da ordem.")

        time.sleep(segundos_espera_por_tentativa)

        detalhe_ordem = consultar_ordem(order_id)
        status, filled_amount, amount = obter_status_e_preenchimento(detalhe_ordem)

        historico.append({
            "tipo": "LIMIT",
            "order_id": order_id,
            "odd_tentada": odd,
            "status": status,
            "filled_amount": filled_amount,
            "amount": amount
        })

        if status == "FILLED":
            return {
                "sucesso": True,
                "tipo_execucao": "LIMIT",
                "order_id": order_id,
                "odd_executada": odd,
                "status_final": status,
                "historico": historico,
                "resposta_final": detalhe_ordem
            }

        if status in ["OPEN", "PARTIALLY_FILLED"]:
            cancelar_ordem(order_id)

            if filled_amount > 0:
                return {
                    "sucesso": True,
                    "tipo_execucao": "LIMIT_PARTIAL",
                    "order_id": order_id,
                    "odd_executada": odd,
                    "status_final": status,
                    "historico": historico,
                    "resposta_final": detalhe_ordem
                }

    if usar_market_no_final:
        resposta_market = criar_ordem_market_buy(
            selection_id=selection_id,
            valor_total=valor_total
        )

        data_market = resposta_market.get("data", {})
        order_id_market = data_market.get("orderId") or data_market.get("id")

        historico.append({
            "tipo": "MARKET",
            "order_id": order_id_market,
            "odd_tentada": None,
            "status": "ENVIADA"
        })

        return {
            "sucesso": True,
            "tipo_execucao": "MARKET",
            "order_id": order_id_market,
            "odd_executada": None,
            "status_final": "ENVIADA",
            "historico": historico,
            "resposta_final": resposta_market
        }

    return {
        "sucesso": False,
        "tipo_execucao": None,
        "order_id": None,
        "odd_executada": None,
        "status_final": "NAO_EXECUTOU",
        "historico": historico,
        "resposta_final": None
    }

def montar_odds_tentativa(nome_mercado, confianca):
    config = CONFIG_ENTRADA_PASSAGENS.get(nome_mercado)

    if not config:
        return [2.20, 2.05, 2.00, 1.90]

    odd_minima_base = config["odd_minima_base"]
    degraus = config["degrau_odds"]

    ajuste_confianca = 0.0

    if confianca >= 0.75:
        ajuste_confianca = -0.10
    elif confianca >= 0.70:
        ajuste_confianca = -0.05
    elif confianca >= 0.66:
        ajuste_confianca = 0.00
    else:
        ajuste_confianca = 0.10

    odd_centro = round(odd_minima_base + ajuste_confianca, 2)

    odds = [round(odd_centro + degrau, 2) for degrau in degraus]
    odds = sorted(set(odds), reverse=True)

    return odds

def processar_ciclo_trade():
    global market_id_em_andamento, order_id_em_andamento

    saldo_antes = obter_saldo()

    try:
        print("SALDO:", saldo_antes)

        mercado = obter_mercado_passagens_aberto()

        if not mercado:
            print("Nenhum mercado aberto encontrado")
            return

        market_id = mercado["id"]
        nome_mercado = mercado.get("title", "Rua (4m 40s): quantas passagens?")

        if market_id_em_andamento is not None and market_id != market_id_em_andamento:
            print("Novo mercado detectado. Liberando trava do mercado anterior.")
            market_id_em_andamento = None
            order_id_em_andamento = None

        if market_id_em_andamento == market_id:
            print("Já existe ordem em acompanhamento para esse mercado. Não enviar nova.")
            return

        config_entrada = CONFIG_ENTRADA_PASSAGENS.get(nome_mercado)

        if not config_entrada:
            print(f"Mercado sem configuração de entrada: {nome_mercado}")
            return

        if not pode_enviar_ordem(mercado):
            print("Não apostar, não pode enviar ordem")
            return

        if not pode_apostar_no_primeiro_minuto(mercado):
            print("Não apostar, mercado já passou do primeiro minuto de abertura")
            return

        threshold_confianca = config_entrada["threshold_confianca"]

        resultado = prever_aposta(mercado, threshold_confianca)

        print({
            "mercado": nome_mercado,
            "confianca": resultado["confianca"],
            "apostar": resultado["apostar"],
            "previsao": resultado["previsao"],
            "threshold": threshold_confianca
        })

        if not resultado["apostar"]:
            print("Não apostar, resultado da previsão não indica aposta")
            return

        selection_id = obter_selection_id_da_previsao(mercado, resultado)

        odds_tentativa = montar_odds_tentativa(
            nome_mercado=nome_mercado,
            confianca=resultado["confianca"]
        )

        print({
            "mercado": nome_mercado,
            "confianca": resultado["confianca"],
            "threshold": threshold_confianca,
            "odds_tentativa": odds_tentativa
        })

        resposta_criacao_ordem = executar_entrada_escalonada(
            selection_id=selection_id,
            valor_total=1.0,
            odds_tentativa=odds_tentativa,
            segundos_espera_por_tentativa=3,
            usar_market_no_final=False
        )

        print(resposta_criacao_ordem)

        if resposta_criacao_ordem["sucesso"]:
            market_id_em_andamento = market_id
            order_id_em_andamento = resposta_criacao_ordem["order_id"]

    except Exception as e:
        print("Erro no ciclo:", str(e))

if __name__ == "__main__":
    schedule.every(5).seconds.do(processar_ciclo_trade)
    processar_ciclo_trade()

    while True:
        schedule.run_pending()

        if schedule.idle_seconds() is None:
            break

        time.sleep(1)

    print("Scheduler encerrado.")