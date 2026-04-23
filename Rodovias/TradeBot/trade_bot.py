import os
import re
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
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Utils import ExecucaoLog, LogService, OrdemLog, PrevisaoLog, ResultadoLog

PASTA_MODELOS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "modelos_rodovias"
)

# Verificação de segurança (opcional, mas ajuda muito no log)
if not os.path.exists(PASTA_MODELOS):
    print(f"AVISO: A pasta de modelos não foi encontrada em: {PASTA_MODELOS}")

ultimo_status_logado = None

CATALOGO_RODOVIAS = {
    "Rodovia Arão Sahm, KM 95 — Bragança Paulista (SP).": {
        "pasta": os.path.join(
            PASTA_MODELOS, "rodovia_arao_sahm_km_95_braganca_paulista_sp_rf_regimes"
        ),
        "modelo": os.path.join(
            PASTA_MODELOS,
            "rodovia_arao_sahm_km_95_braganca_paulista_sp_rf_regimes",
            "modelo.joblib",
        ),
        "metadata_path": os.path.join(
            PASTA_MODELOS,
            "rodovia_arao_sahm_km_95_braganca_paulista_sp_rf_regimes",
            "metadata.json",
        ),
    },
    "Floriano Rodrigues Pinheiro, KM 26 — Pindamonhangaba (SP).": {
        "pasta": os.path.join(
            PASTA_MODELOS,
            "floriano_rodrigues_pinheiro_km_26_pindamonhangaba_sp_rf_regimes",
        ),
        "modelo": os.path.join(
            PASTA_MODELOS,
            "floriano_rodrigues_pinheiro_km_26_pindamonhangaba_sp_rf_regimes",
            "modelo.joblib",
        ),
        "metadata_path": os.path.join(
            PASTA_MODELOS,
            "floriano_rodrigues_pinheiro_km_26_pindamonhangaba_sp_rf_regimes",
            "metadata.json",
        ),
    },
    "Doutor Manoel Hyppolito Rego, KM 83 — Caraguatatuba (SP).": {
        "pasta": os.path.join(
            PASTA_MODELOS,
            "doutor_manoel_hyppolito_rego_km_83_caraguatatuba_sp_rf_regimes",
        ),
        "modelo": os.path.join(
            PASTA_MODELOS,
            "doutor_manoel_hyppolito_rego_km_83_caraguatatuba_sp_rf_regimes",
            "modelo.joblib",
        ),
        "metadata_path": os.path.join(
            PASTA_MODELOS,
            "doutor_manoel_hyppolito_rego_km_83_caraguatatuba_sp_rf_regimes",
            "metadata.json",
        ),
    },
}

CONFIG_ENTRADA_RODOVIA = {
    "Rodovia Arão Sahm, KM 95 — Bragança Paulista (SP).": {
        "threshold_confianca": 0.60,
        "odd_minima_base": 2.05,
        "degrau_odds": [0.25, 0.10, 0.00, -0.10],
    },
    "Doutor Manoel Hyppolito Rego, KM 83 — Caraguatatuba (SP).": {
        "threshold_confianca": 0.62,
        "odd_minima_base": 1.95,
        "degrau_odds": [0.25, 0.10, 0.00, -0.10],
    },
    "Floriano Rodrigues Pinheiro, KM 26 — Pindamonhangaba (SP).": {
        "threshold_confianca": 0.60,
        "odd_minima_base": 2.05,
        "degrau_odds": [0.25, 0.10, 0.00, -0.10],
    },
}

log_service = LogService()


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

coordenadas_rodovias = {
    ("Rodovia Arão Sahm", 95): {"latitude": -22.9256, "longitude": -46.5529},
    ("Doutor Manoel Hyppolito Rego", 83): {"latitude": -23.5663, "longitude": -45.2793},
    ("Doutor Manoel Hyppolito Rego", 83.5): {
        "latitude": -23.5663,
        "longitude": -45.2793,
    },
    ("Doutor Manoel Hyppolito Rego", 110): {
        "latitude": -23.6980,
        "longitude": -45.4395,
    },
    ("Doutor Manoel Hyppolito Rego", 110.8): {
        "latitude": -23.6980,
        "longitude": -45.4395,
    },
    ("Floriano Rodrigues Pinheiro", 46): {"latitude": -22.7561, "longitude": -45.6102},
    ("Floriano Rodrigues Pinheiro", 26): {"latitude": -22.8640, "longitude": -45.6025},
    ("Floriano Rodrigues Pinheiro", 26.5): {
        "latitude": -22.8640,
        "longitude": -45.6025,
    },
}


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


def obter_saldo():

    token = base64.b64encode(f"{api_key}:{api_secret}".encode()).decode()

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

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


def obter_mercado_carros_aberto():
    url = "https://app.palpita.io/api/v1/markets"
    params = {
        "page": 1,
        "limit": 100,
        "search": "Rodovia (5 minutos): quantos carros?",
        "orderBy": "closesAt",
        "orderDirection": "ASC",
    }
    headers = {"Accept": "application/json"}

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

    closes_betting_at = mercado.get("closesBettingAt")
    if not closes_betting_at:
        return False

    agora = datetime.now().astimezone()
    limite = datetime.fromisoformat(closes_betting_at)

    return agora <= limite


def extrair_rodovia_km(rodovia_identificacao):
    texto = str(rodovia_identificacao)

    if "KM" not in texto.upper():
        return None, None

    partes = texto.split(", KM")
    nome = partes[0].strip()

    km_texto = (
        partes[1].split("—")[0].strip().replace(",", ".") if len(partes) > 1 else ""
    )
    try:
        km = float(km_texto)
    except Exception:
        km = None

    return nome, km


def buscar_coordenadas_por_rodovia(rodovia_identificacao):
    nome, km = extrair_rodovia_km(rodovia_identificacao)

    if nome is None or km is None:
        return None, None

    chave_exata = (nome, km)
    if chave_exata in coordenadas_rodovias:
        item = coordenadas_rodovias[chave_exata]
        return item["latitude"], item["longitude"]

    for (nome_base, km_base), item in coordenadas_rodovias.items():
        if nome_base == nome and abs(float(km_base) - float(km)) < 0.01:
            return item["latitude"], item["longitude"]

    return None, None


def buscar_clima_atual(latitude, longitude):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,relative_humidity_2m,precipitation,cloud_cover",
        "timezone": "America/Sao_Paulo",
    }

    response = session.get(url, params=params, timeout=60)
    response.raise_for_status()
    dados = response.json()

    current = dados.get("current", {})

    temperatura_2m = current.get("temperature_2m")
    umidade_relativa = current.get("relative_humidity_2m")
    chuva_mm = current.get("precipitation")
    cobertura_nuvens = current.get("cloud_cover")

    estava_chovendo = 1 if chuva_mm is not None and float(chuva_mm) > 0 else 0

    return {
        "temperatura_2m": float(temperatura_2m) if temperatura_2m is not None else 0.0,
        "umidade_relativa": (
            float(umidade_relativa) if umidade_relativa is not None else 0.0
        ),
        "chuva_mm": float(chuva_mm) if chuva_mm is not None else 0.0,
        "cobertura_nuvens": (
            float(cobertura_nuvens) if cobertura_nuvens is not None else 0.0
        ),
        "estava_chovendo": float(estava_chovendo),
    }


def carregar_catalogo_modelos():
    catalogo = {}

    for rodovia, item in CATALOGO_RODOVIAS.items():
        caminho_metadata = item["metadata_path"]
        caminho_modelo = item["modelo"]
        caminho_pasta = item["pasta"]

        if not os.path.exists(caminho_metadata) or not os.path.exists(caminho_modelo):
            continue

        with open(caminho_metadata, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        catalogo[rodovia] = {
            "pasta": caminho_pasta,
            "modelo": caminho_modelo,
            "metadata": metadata,
        }

    return catalogo


def prever_aposta(
    catalogo, rodovia_identificacao, meta_referencia, threshold_confianca
):

    if rodovia_identificacao not in catalogo:
        raise ValueError(
            f"Rodovia não encontrada no catálogo de modelos: {rodovia_identificacao}"
        )

    item = catalogo[rodovia_identificacao]
    bundle = joblib.load(item["modelo"])

    if not isinstance(bundle, dict) or "modelo" not in bundle:
        raise ValueError(
            "Esse modelo está em formato antigo/incompatível com o novo modelo."
        )

    modelo = bundle["modelo"]
    features = bundle.get("features", [])

    agora = datetime.now()

    hora = agora.hour
    minuto = agora.minute
    dia_semana = agora.weekday()
    mes = agora.month
    hora_decimal = hora + (minuto / 60.0)
    fim_semana = 1 if dia_semana in [5, 6] else 0

    latitude, longitude = buscar_coordenadas_por_rodovia(rodovia_identificacao)
    if latitude is None or longitude is None:
        raise ValueError(
            "Não foi possível localizar coordenadas da rodovia para consultar o clima."
        )

    clima = buscar_clima_atual(latitude, longitude)

    entrada_dict = {
        "meta_num": float(meta_referencia),
        "fim_semana": fim_semana,
        "hora": hora,
        "minuto": minuto,
        "hora_decimal": hora_decimal,
        "dia_semana": dia_semana,
        "mes": mes,
        "hora_sin": np.sin(2 * np.pi * hora_decimal / 24),
        "hora_cos": np.cos(2 * np.pi * hora_decimal / 24),
        "dia_semana_sin": np.sin(2 * np.pi * dia_semana / 7),
        "dia_semana_cos": np.cos(2 * np.pi * dia_semana / 7),
        "mes_sin": np.sin(2 * np.pi * (mes - 1) / 12),
        "mes_cos": np.cos(2 * np.pi * (mes - 1) / 12),
        "temperatura_2m": clima["temperatura_2m"],
        "umidade_relativa": clima["umidade_relativa"],
        "chuva_mm": clima["chuva_mm"],
        "cobertura_nuvens": clima["cobertura_nuvens"],
        "estava_chovendo": clima["estava_chovendo"],
    }

    entrada_dict["eh_pico_manha"] = int(5 <= hora <= 8)
    entrada_dict["eh_pico_tarde"] = int(16 <= hora <= 18)
    entrada_dict["eh_sexta"] = int(dia_semana == 4)
    entrada_dict["eh_domingo"] = int(dia_semana == 6)
    entrada_dict["eh_sabado"] = int(dia_semana == 5)
    entrada_dict["eh_dia_util"] = int(dia_semana <= 4)

    if 0 <= hora_decimal < 5:
        bloco_dia = "madrugada"
    elif 5 <= hora_decimal < 8:
        bloco_dia = "pico_manha"
    elif 8 <= hora_decimal < 11:
        bloco_dia = "manha"
    elif 11 <= hora_decimal < 14:
        bloco_dia = "almoco"
    elif 14 <= hora_decimal < 18:
        bloco_dia = "tarde_pico"
    elif 18 <= hora_decimal < 22:
        bloco_dia = "noite"
    else:
        bloco_dia = "late_night"

    mapa_bloco = {
        "madrugada": 0,
        "pico_manha": 1,
        "manha": 2,
        "almoco": 3,
        "tarde_pico": 4,
        "noite": 5,
        "late_night": 6,
        "outro": 7,
    }

    entrada_dict["bloco_dia_cod"] = mapa_bloco.get(bloco_dia, 7)

    for nome_bloco in [
        "bloco_madrugada",
        "bloco_pico_manha",
        "bloco_manha",
        "bloco_almoco",
        "bloco_tarde_pico",
        "bloco_noite",
        "bloco_late_night",
        "bloco_outro",
    ]:
        entrada_dict[nome_bloco] = 0

    chave_bloco = f"bloco_{bloco_dia}"
    if chave_bloco in entrada_dict:
        entrada_dict[chave_bloco] = 1

    entrada_dict["interacao_meta_hora_sin"] = (
        entrada_dict["meta_num"] * entrada_dict["hora_sin"]
    )
    entrada_dict["interacao_meta_hora_cos"] = (
        entrada_dict["meta_num"] * entrada_dict["hora_cos"]
    )
    entrada_dict["interacao_meta_temperatura"] = (
        entrada_dict["meta_num"] * entrada_dict["temperatura_2m"]
    )
    entrada_dict["interacao_meta_umidade"] = (
        entrada_dict["meta_num"] * entrada_dict["umidade_relativa"]
    )

    colunas_faltantes = [col for col in features if col not in entrada_dict]
    if colunas_faltantes:
        raise ValueError(
            "A entrada de previsão não possui todas as features esperadas pelo modelo. "
            f"Faltando: {colunas_faltantes}"
        )

    entrada = pd.DataFrame([entrada_dict])[features]

    probas = modelo.predict_proba(entrada)[0]
    classes_modelo = list(modelo.named_steps["model"].classes_)
    prob_por_classe = {
        int(classe): float(prob) for classe, prob in zip(classes_modelo, probas)
    }

    prob_ate = prob_por_classe.get(0, 0.0)
    prob_mais = prob_por_classe.get(1, 0.0)

    if prob_mais >= prob_ate:
        previsao = f"Mais de {meta_referencia}"
        confianca = prob_mais
    else:
        previsao = f"Até {meta_referencia}"
        confianca = prob_ate

    apostar = confianca >= threshold_confianca

    return {
        "rodovia": rodovia_identificacao,
        "meta_referencia": float(meta_referencia),
        "horario_execucao": agora.strftime("%d/%m/%Y %H:%M:%S"),
        "previsao": previsao,
        "prob_mais": prob_mais,
        "prob_ate": prob_ate,
        "confianca": confianca,
        "apostar": apostar,
        "threshold_utilizado": float(threshold_confianca),
    }


def prever_aposta_por_mercado(catalogo, mercado, threshold_confianca):
    rodovia_identificacao = extrair_rodovia_do_mercado(mercado)

    if not rodovia_identificacao:
        raise ValueError("Não foi possível extrair a rodovia do mercado.")

    meta_referencia = None
    for selection in mercado.get("selections", []):
        label = str(selection.get("label", "")).strip()
        if label.startswith("Mais de "):
            meta_referencia = float(label.replace("Mais de ", "").strip())
            break
        if label.startswith("Até "):
            meta_referencia = float(label.replace("Até ", "").strip())
            break

    if meta_referencia is None:
        raise ValueError("Não foi possível extrair a meta do mercado.")

    return prever_aposta(
        catalogo=catalogo,
        rodovia_identificacao=rodovia_identificacao,
        meta_referencia=meta_referencia,
        threshold_confianca=threshold_confianca,
    )


def obter_selection_id_da_previsao(mercado, resultado):
    previsao = resultado["previsao"]
    meta_int = int(float(resultado["meta_referencia"]))

    if previsao.startswith("Mais de"):
        label_esperada = f"Mais de {meta_int}"
    else:
        label_esperada = f"Até {meta_int}"

    selection = next(
        (s for s in mercado.get("selections", []) if s.get("label") == label_esperada),
        None,
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
        "Content-Type": "application/json",
    }

    payload = {
        "selectionId": int(selection_id),
        "side": "BUY",
        "type": "LIMIT",
        "price": f"{price:.2f}",
        "amount": f"{amount:.4f}",
    }

    response = requests.post(
        "https://app.palpita.io/api/v1/orders",
        headers=headers,
        json=payload,
        timeout=30,
    )

    print("PAYLOAD:", payload)
    print("STATUS:", response.status_code)
    print("RESPOSTA:", response.text)

    response.raise_for_status()
    return response.json()


def consultar_ordem(order_id):
    token = base64.b64encode(f"{api_key}:{api_secret}".encode()).decode()

    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}

    response = requests.get(
        f"https://app.palpita.io/api/v1/orders/{order_id}", headers=headers, timeout=30
    )

    print("CONSULTAR ORDEM STATUS:", response.status_code)
    print("CONSULTAR ORDEM RESPOSTA:", response.text)

    response.raise_for_status()
    return response.json()


def cancelar_ordem(order_id):
    token = base64.b64encode(f"{api_key}:{api_secret}".encode()).decode()

    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}

    response = requests.delete(
        f"https://app.palpita.io/api/v1/orders/{order_id}", headers=headers, timeout=30
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
        "Content-Type": "application/json",
    }

    payload = {
        "selectionId": int(selection_id),
        "side": "BUY",
        "type": "MARKET",
        "total": f"{valor_total:.2f}",
    }

    response = requests.post(
        "https://app.palpita.io/api/v1/orders",
        headers=headers,
        json=payload,
        timeout=30,
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
    usar_market_no_final=False,
):
    historico = []

    for odd in odds_tentativa:
        resposta_ordem = criar_ordem_limit_buy(
            selection_id=selection_id, valor_total=valor_total, odd_minima=odd
        )

        data_ordem = resposta_ordem.get("data", {})
        order_id = data_ordem.get("orderId") or data_ordem.get("id")

        if not order_id:
            raise ValueError("A API não retornou orderId na criação da ordem.")

        time.sleep(segundos_espera_por_tentativa)

        detalhe_ordem = consultar_ordem(order_id)
        status, filled_amount, amount = obter_status_e_preenchimento(detalhe_ordem)

        historico.append(
            {
                "tipo": "LIMIT",
                "order_id": order_id,
                "odd_tentada": odd,
                "status": status,
                "filled_amount": filled_amount,
                "amount": amount,
            }
        )

        if status == "FILLED":
            return {
                "sucesso": True,
                "tipo_execucao": "LIMIT",
                "order_id": order_id,
                "odd_executada": odd,
                "status_final": status,
                "historico": historico,
                "resposta_final": detalhe_ordem,
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
                    "resposta_final": detalhe_ordem,
                }

    if usar_market_no_final:
        resposta_market = criar_ordem_market_buy(
            selection_id=selection_id, valor_total=valor_total
        )

        data_market = resposta_market.get("data", {})
        order_id_market = data_market.get("orderId") or data_market.get("id")

        historico.append(
            {
                "tipo": "MARKET",
                "order_id": order_id_market,
                "odd_tentada": None,
                "status": "ENVIADA",
            }
        )

        return {
            "sucesso": True,
            "tipo_execucao": "MARKET",
            "order_id": order_id_market,
            "odd_executada": None,
            "status_final": "ENVIADA",
            "historico": historico,
            "resposta_final": resposta_market,
        }

    return {
        "sucesso": False,
        "tipo_execucao": None,
        "order_id": None,
        "odd_executada": None,
        "status_final": "NAO_EXECUTOU",
        "historico": historico,
        "resposta_final": None,
    }


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


def extrair_rodovia_do_mercado(mercado):
    descricao = str(mercado.get("description", "")).strip()

    if not descricao:
        return None

    for linha in descricao.splitlines():
        linha_limpa = linha.strip().lstrip("•").strip()

        if "KM" in linha_limpa and "—" in linha_limpa and "(SP)" in linha_limpa:
            return linha_limpa

    return None


def montar_odds_tentativa(rodovia, confianca):
    config = CONFIG_ENTRADA_RODOVIA.get(rodovia)

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


ultimo_log = {}


def log_unico(chave, mensagem):
    global ultimo_log

    if ultimo_log.get(chave) != mensagem:
        print(mensagem)
        ultimo_log[chave] = mensagem


def registrar_execucao(rodovia, etapa, status, mensagem, market_id="", nome_metodo=""):
    try:
        log_service.registrar_execucao(
            ExecucaoLog(
                data_hora=LogService.agora_str(),
                rodovia=str(rodovia or ""),
                etapa=etapa,
                status=status,
                mensagem=mensagem,
                market_id=str(market_id or ""),
                nome_metodo=nome_metodo,
                tempo_execucao_ms=0,
            )
        )
    except Exception as erro_log:
        print(f"Falha ao registrar log de execução: {erro_log}")


def processar_ciclo_trade():
    global market_id_em_andamento, order_id_em_andamento

    saldo_antes = obter_saldo()

    try:
        catalogo = carregar_catalogo_modelos()

        log_unico("saldo", f"SALDO: {saldo_antes}")

        mercado = obter_mercado_carros_aberto()

        if not mercado:
            print("status", "Nenhum mercado aberto encontrado")
            registrar_execucao(
                rodovia="",
                etapa="buscar_mercado",
                status="SEM_MERCADO",
                mensagem="Nenhum mercado aberto encontrado",
                nome_metodo="processar_ciclo_trade",
            )
            return

        market_id = mercado["id"]

        if market_id_em_andamento is not None and market_id != market_id_em_andamento:
            print(
                "status", "Novo mercado detectado. Liberando trava do mercado anterior."
            )
            market_id_em_andamento = None
            order_id_em_andamento = None

        if market_id_em_andamento == market_id:
            print(
                "status",
                "Já existe ordem em acompanhamento para esse mercado. Não enviar nova.",
            )
            registrar_execucao(
                rodovia="",
                etapa="controle_mercado",
                status="EM_ANDAMENTO",
                mensagem="Já existe ordem em acompanhamento para esse mercado",
                market_id=market_id,
                nome_metodo="processar_ciclo_trade",
            )
            return

        rodovia_mercado = extrair_rodovia_do_mercado(mercado)

        if not rodovia_mercado:
            log_unico("status", "Não foi possível identificar a rodovia do mercado")
            registrar_execucao(
                rodovia="",
                etapa="extrair_rodovia",
                status="ERRO",
                mensagem="Não foi possível identificar a rodovia do mercado",
                market_id=market_id,
                nome_metodo="processar_ciclo_trade",
            )
            return

        if rodovia_mercado not in catalogo:
            log_unico("status", f"Rodovia sem modelo cadastrado: {rodovia_mercado}")
            registrar_execucao(
                rodovia=rodovia_mercado,
                etapa="validar_modelo",
                status="SEM_MODELO",
                mensagem=f"Rodovia sem modelo cadastrado: {rodovia_mercado}",
                market_id=market_id,
                nome_metodo="processar_ciclo_trade",
            )
            return

        config_rodovia = CONFIG_ENTRADA_RODOVIA.get(rodovia_mercado)

        if not config_rodovia:
            log_unico(
                "status", f"Rodovia sem configuração de entrada: {rodovia_mercado}"
            )
            registrar_execucao(
                rodovia=rodovia_mercado,
                etapa="validar_configuracao",
                status="SEM_CONFIG",
                mensagem=f"Rodovia sem configuração de entrada: {rodovia_mercado}",
                market_id=market_id,
                nome_metodo="processar_ciclo_trade",
            )
            return

        if not pode_enviar_ordem(mercado):
            log_unico("status", "Não apostar, não pode enviar ordem")
            registrar_execucao(
                rodovia=rodovia_mercado,
                etapa="validar_janela_aposta",
                status="BLOQUEADO",
                mensagem="Não pode enviar ordem para esse mercado",
                market_id=market_id,
                nome_metodo="processar_ciclo_trade",
            )
            return

        if not pode_apostar_no_primeiro_minuto(mercado):
            log_unico(
                "status",
                "Não apostar, mercado já passou do primeiro minuto de abertura",
            )
            registrar_execucao(
                rodovia=rodovia_mercado,
                etapa="validar_primeiro_minuto",
                status="FORA_JANELA",
                mensagem="Mercado já passou do primeiro minuto de abertura",
                market_id=market_id,
                nome_metodo="processar_ciclo_trade",
            )
            return

        threshold_confianca = config_rodovia["threshold_confianca"]

        resultado = prever_aposta_por_mercado(catalogo, mercado, threshold_confianca)

        log_service.registrar_previsao(
            PrevisaoLog(
                data_hora=LogService.agora_str(),
                rodovia=rodovia_mercado,
                market_id=str(market_id),
                selection_id=None,
                nome_modelo="rf_producao_friendly_sem_lags",
                classe_prevista=resultado["previsao"],
                confianca=float(resultado["confianca"]),
                threshold=float(threshold_confianca),
                meta_referencia=str(resultado["meta_referencia"]),
                odd_minima_aceita=float(montar_odds_tentativa(rodovia_mercado, resultado["confianca"])[0]),
            )
        )

        # 🔒 NOVA REGRA AQUI
        # if not resultado["previsao"].startswith("Mais de"):
        #    log_unico("status", "Não apostar, previsão não é 'Mais de'")
        #    return

        log_unico(
            "previsao",
            json.dumps(
                {
                    "rodovia": rodovia_mercado,
                    "confianca": resultado["confianca"],
                    "apostar": resultado["apostar"],
                    "previsao": resultado["previsao"],
                    "threshold": threshold_confianca,
                },
                sort_keys=True,
            ),
        )

        if not resultado["apostar"]:
            log_unico("status", "Não apostar, resultado da previsão não indica aposta")
            log_service.registrar_resultado(
                ResultadoLog(
                    data_hora=LogService.agora_str(),
                    rodovia=rodovia_mercado,
                    market_id=str(market_id),
                    selection_id="",
                    classe_prevista=resultado["previsao"],
                    confianca=float(resultado["confianca"]),
                    aposta_realizada=False,
                    direcao_aposta="",
                    valor_observado_depois=0.0,
                    meta_ou_linha_mercado=str(resultado["meta_referencia"]),
                    acertou_previsao=False,
                    aposta_ganha=False,
                    lucro_prejuizo=0.0,
                )
            )
            return

        selection_id = obter_selection_id_da_previsao(mercado, resultado)

        odds_tentativa = montar_odds_tentativa(
            rodovia=rodovia_mercado, confianca=resultado["confianca"]
        )

        log_unico(
            "odds",
            json.dumps(
                {
                    "rodovia": rodovia_mercado,
                    "confianca": resultado["confianca"],
                    "threshold": threshold_confianca,
                    "odds_tentativa": odds_tentativa,
                },
                sort_keys=True,
            ),
        )

        resposta_criacao_ordem = executar_entrada_escalonada(
            selection_id=selection_id,
            valor_total=1,
            odds_tentativa=odds_tentativa,
            segundos_espera_por_tentativa=3,
            usar_market_no_final=False,
        )

        log_unico("ordem", json.dumps(resposta_criacao_ordem, sort_keys=True))
        log_service.registrar_ordem(
            OrdemLog(
                data_hora=LogService.agora_str(),
                rodovia=rodovia_mercado,
                market_id=str(market_id),
                selection_id=str(selection_id),
                tipo_ordem=str(resposta_criacao_ordem.get("tipo_execucao") or ""),
                direcao_aposta="BUY",
                stake=1.0,
                odd_solicitada=float(odds_tentativa[0]),
                ordem_enviada=True,
                ordem_executada=bool(resposta_criacao_ordem.get("sucesso")),
                status_ordem=str(resposta_criacao_ordem.get("status_final") or ""),
                order_id=str(resposta_criacao_ordem.get("order_id") or ""),
            )
        )

        if resposta_criacao_ordem["sucesso"]:
            market_id_em_andamento = market_id
            order_id_em_andamento = resposta_criacao_ordem["order_id"]

        registrar_execucao(
            rodovia=rodovia_mercado,
            etapa="ciclo_finalizado",
            status="SUCESSO" if resposta_criacao_ordem["sucesso"] else "SEM_ENTRADA",
            mensagem="Ciclo de trade processado",
            market_id=market_id,
            nome_metodo="processar_ciclo_trade",
        )

    except Exception as e:
        log_unico("erro", f"Erro no ciclo: {str(e)}")
        registrar_execucao(
            rodovia="",
            etapa="erro_ciclo",
            status="ERRO",
            mensagem=str(e),
            market_id=str(market_id_em_andamento or ""),
            nome_metodo="processar_ciclo_trade",
        )


if __name__ == "__main__":

    # schedule.every(5).seconds.until("22:00").do(processar_ciclo_trade)
    schedule.every(3).seconds.do(processar_ciclo_trade)
    processar_ciclo_trade()

    while True:
        schedule.run_pending()

        if schedule.idle_seconds() is None:
            break

        time.sleep(1)

    print("Scheduler encerrado.")
