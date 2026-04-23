import os
import re
import json
import joblib
import numpy as np
import pandas as pd
import customtkinter as ctk
from datetime import datetime
from tkinter import StringVar
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


PASTA_MODELOS = os.path.join(os.path.dirname(os.path.abspath(__file__)),"modelos_rodovias")

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

def extrair_rodovia_km(rodovia_identificacao):
    texto = str(rodovia_identificacao)

    if "KM" not in texto.upper():
        return None, None

    partes = texto.split(", KM")
    nome = partes[0].strip()

    km_texto = partes[1].split("—")[0].strip().replace(",", ".") if len(partes) > 1 else ""
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
        "timezone": "America/Sao_Paulo"
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
        "umidade_relativa": float(umidade_relativa) if umidade_relativa is not None else 0.0,
        "chuva_mm": float(chuva_mm) if chuva_mm is not None else 0.0,
        "cobertura_nuvens": float(cobertura_nuvens) if cobertura_nuvens is not None else 0.0,
        "estava_chovendo": float(estava_chovendo)
    }

def normalizar_nome_arquivo(texto):
    texto = str(texto).strip().lower()
    texto = re.sub(r"[^\w\s-]", "", texto, flags=re.UNICODE)
    texto = re.sub(r"[\s]+", "_", texto)
    return texto


def carregar_catalogo_modelos(pasta_modelos):
    catalogo = {}

    if not os.path.exists(pasta_modelos):
        return catalogo

    for nome_pasta in os.listdir(pasta_modelos):
        caminho_pasta = os.path.join(pasta_modelos, nome_pasta)

        if not os.path.isdir(caminho_pasta):
            continue

        caminho_metadata = os.path.join(caminho_pasta, "metadata.json")
        caminho_modelo = os.path.join(caminho_pasta, "modelo.joblib")

        if not os.path.exists(caminho_metadata) or not os.path.exists(caminho_modelo):
            continue

        with open(caminho_metadata, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        rodovia = metadata.get("rodovia_identificacao")
        if not rodovia:
            continue

        catalogo[rodovia] = {
            "pasta": caminho_pasta,
            "modelo": caminho_modelo,
            "metadata": metadata
        }

    return dict(sorted(catalogo.items(), key=lambda x: x[0]))


def prever_aposta(catalogo, rodovia_identificacao, meta_referencia, threshold_confianca=0.60):
    if rodovia_identificacao not in catalogo:
        raise ValueError("Rodovia não encontrada no catálogo de modelos.")

    item = catalogo[rodovia_identificacao]
    metadata = item["metadata"]
    bundle = joblib.load(item["modelo"])

    if not isinstance(bundle, dict) or "modelo" not in bundle:
        raise ValueError("Esse modelo está em formato antigo/incompatível com o novo modelo.")

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
        raise ValueError("Não foi possível localizar coordenadas da rodovia para consultar o clima.")

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
        "estava_chovendo": clima["estava_chovendo"]
    }

    entrada_dict["eh_pico_manha"] = int(hora >= 5 and hora <= 8)
    entrada_dict["eh_pico_tarde"] = int(hora >= 16 and hora <= 18)
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
        "outro": 7
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
        "bloco_outro"
    ]:
        entrada_dict[nome_bloco] = 0

    chave_bloco = f"bloco_{bloco_dia}"
    if chave_bloco in entrada_dict:
        entrada_dict[chave_bloco] = 1

    entrada_dict["interacao_meta_hora_sin"] = entrada_dict["meta_num"] * entrada_dict["hora_sin"]
    entrada_dict["interacao_meta_hora_cos"] = entrada_dict["meta_num"] * entrada_dict["hora_cos"]
    entrada_dict["interacao_meta_temperatura"] = entrada_dict["meta_num"] * entrada_dict["temperatura_2m"]
    entrada_dict["interacao_meta_umidade"] = entrada_dict["meta_num"] * entrada_dict["umidade_relativa"]

    for col in features:
        if col not in entrada_dict:
            entrada_dict[col] = np.nan

    entrada = pd.DataFrame([entrada_dict])[features]

    probas = modelo.predict_proba(entrada)[0]
    classes_modelo = list(modelo.named_steps["model"].classes_)
    prob_por_classe = {int(classe): float(prob) for classe, prob in zip(classes_modelo, probas)}

    prob_ate = prob_por_classe.get(0, 0.0)
    prob_mais = prob_por_classe.get(1, 0.0)

    if prob_mais >= prob_ate:
        previsao = f"Mais de {meta_referencia}"
        confianca = prob_mais
    else:
        previsao = f"Até {meta_referencia}"
        confianca = prob_ate

    apostar = confianca >= threshold_confianca
    metricas = metadata.get("metricas_teste", {})

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
        "metricas_modelo": {
            "accuracy": metricas.get("accuracy"),
            "auc": metricas.get("auc"),
            "precision_mais": metricas.get("precision_mais"),
            "recall_mais": metricas.get("recall_mais"),
            "f1_mais": metricas.get("f1_mais"),
            "baseline_majoritaria": metricas.get("baseline_majoritaria"),
            "ganho_vs_baseline": metricas.get("ganho_vs_baseline")
        }
    }


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Previsão de Rodovias")
        self.geometry("1180x760")
        self.minsize(1080, 700)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.catalogo = carregar_catalogo_modelos(PASTA_MODELOS)

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.main = ctk.CTkFrame(self, corner_radius=18)
        self.main.grid(row=0, column=0, sticky="nsew", padx=18, pady=18)
        self.main.grid_columnconfigure(0, weight=1)
        self.main.grid_columnconfigure(1, weight=1)
        self.main.grid_rowconfigure(2, weight=1)

        self._criar_header()
        self._criar_formulario()
        self._criar_resultado()

    def _criar_header(self):
        header = ctk.CTkFrame(self.main, corner_radius=16)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=16, pady=(16, 10))
        header.grid_columnconfigure(0, weight=1)

        titulo = ctk.CTkLabel(
            header,
            text="Previsão de Aposta por Rodovia",
            font=ctk.CTkFont(size=28, weight="bold")
        )
        titulo.grid(row=0, column=0, sticky="w", padx=18, pady=(16, 4))

        subtitulo = ctk.CTkLabel(
            header,
            text="Informe a rodovia e a meta. O sistema usa horário atual automaticamente e mostra a recomendação na tela.",
            font=ctk.CTkFont(size=14)
        )
        subtitulo.grid(row=1, column=0, sticky="w", padx=18, pady=(0, 16))

    def _criar_formulario(self):
        card = ctk.CTkFrame(self.main, corner_radius=16)
        card.grid(row=1, column=0, columnspan=2, sticky="ew", padx=16, pady=10)
        card.grid_columnconfigure(0, weight=1)
        card.grid_columnconfigure(1, weight=1)
        card.grid_columnconfigure(2, weight=1)
        card.grid_columnconfigure(3, weight=0)

        label_rodovia = ctk.CTkLabel(card, text="Rodovia", font=ctk.CTkFont(size=15, weight="bold"))
        label_rodovia.grid(row=0, column=0, sticky="w", padx=(18, 8), pady=(16, 6))

        self.combo_rodovia = ctk.CTkComboBox(
            card,
            values=list(self.catalogo.keys()) if self.catalogo else ["Nenhum modelo encontrado"],
            width=500
        )
        self.combo_rodovia.grid(row=1, column=0, sticky="ew", padx=(18, 8), pady=(0, 16))

        label_meta = ctk.CTkLabel(card, text="Meta de referência", font=ctk.CTkFont(size=15, weight="bold"))
        label_meta.grid(row=0, column=1, sticky="w", padx=8, pady=(16, 6))

        self.entry_meta = ctk.CTkEntry(card, placeholder_text="Ex.: 101")
        self.entry_meta.grid(row=1, column=1, sticky="ew", padx=8, pady=(0, 16))

        label_threshold = ctk.CTkLabel(card, text="Threshold de confiança", font=ctk.CTkFont(size=15, weight="bold"))
        label_threshold.grid(row=0, column=2, sticky="w", padx=8, pady=(16, 6))

        self.entry_threshold = ctk.CTkEntry(card, placeholder_text="Ex.: 0.60")
        self.entry_threshold.insert(0, "0.60")
        self.entry_threshold.grid(row=1, column=2, sticky="ew", padx=8, pady=(0, 16))

        self.btn_prever = ctk.CTkButton(
            card,
            text="Prever",
            width=140,
            height=40,
            command=self.executar_previsao
        )
        self.btn_prever.grid(row=1, column=3, sticky="e", padx=(8, 18), pady=(0, 16))

    def _criar_resultado(self):
        container = ctk.CTkFrame(self.main, corner_radius=16)
        container.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=16, pady=(10, 16))
        container.grid_columnconfigure(0, weight=1)
        container.grid_columnconfigure(1, weight=1)
        container.grid_rowconfigure(1, weight=1)

        self.lbl_status = ctk.CTkLabel(
            container,
            text="Preencha os campos e clique em Prever.",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.lbl_status.grid(row=0, column=0, columnspan=2, sticky="w", padx=18, pady=(18, 12))

        self.card_decisao = ctk.CTkFrame(container, corner_radius=16)
        self.card_decisao.grid(row=1, column=0, sticky="nsew", padx=(18, 9), pady=(0, 18))
        self.card_decisao.grid_columnconfigure(0, weight=1)

        self.card_metricas = ctk.CTkFrame(container, corner_radius=16)
        self.card_metricas.grid(row=1, column=1, sticky="nsew", padx=(9, 18), pady=(0, 18))
        self.card_metricas.grid_columnconfigure(0, weight=1)

        self.lbl_resultado_titulo = ctk.CTkLabel(
            self.card_decisao,
            text="Resultado da previsão",
            font=ctk.CTkFont(size=22, weight="bold")
        )
        self.lbl_resultado_titulo.grid(row=0, column=0, sticky="w", padx=18, pady=(18, 10))

        self.lbl_previsao = ctk.CTkLabel(
            self.card_decisao,
            text="—",
            font=ctk.CTkFont(size=28, weight="bold")
        )
        self.lbl_previsao.grid(row=1, column=0, sticky="w", padx=18, pady=4)

        self.lbl_apostar = ctk.CTkLabel(
            self.card_decisao,
            text="—",
            font=ctk.CTkFont(size=22, weight="bold")
        )
        self.lbl_apostar.grid(row=2, column=0, sticky="w", padx=18, pady=4)

        self.lbl_confianca = ctk.CTkLabel(
            self.card_decisao,
            text="Confiança: —",
            font=ctk.CTkFont(size=16)
        )
        self.lbl_confianca.grid(row=3, column=0, sticky="w", padx=18, pady=4)

        self.lbl_prob_mais = ctk.CTkLabel(
            self.card_decisao,
            text="Probabilidade de Mais: —",
            font=ctk.CTkFont(size=16)
        )
        self.lbl_prob_mais.grid(row=4, column=0, sticky="w", padx=18, pady=4)

        self.lbl_prob_ate = ctk.CTkLabel(
            self.card_decisao,
            text="Probabilidade de Até: —",
            font=ctk.CTkFont(size=16)
        )
        self.lbl_prob_ate.grid(row=5, column=0, sticky="w", padx=18, pady=4)

        self.lbl_horario = ctk.CTkLabel(
            self.card_decisao,
            text="Horário usado: —",
            font=ctk.CTkFont(size=16)
        )
        self.lbl_horario.grid(row=6, column=0, sticky="w", padx=18, pady=4)

        self.lbl_rodovia = ctk.CTkLabel(
            self.card_decisao,
            text="Rodovia: —",
            font=ctk.CTkFont(size=16)
        )
        self.lbl_rodovia.grid(row=7, column=0, sticky="w", padx=18, pady=4)

        self.lbl_meta = ctk.CTkLabel(
            self.card_decisao,
            text="Meta: —",
            font=ctk.CTkFont(size=16)
        )
        self.lbl_meta.grid(row=8, column=0, sticky="w", padx=18, pady=(4, 18))

        self.lbl_metricas_titulo = ctk.CTkLabel(
            self.card_metricas,
            text="Métricas do modelo da rodovia",
            font=ctk.CTkFont(size=22, weight="bold")
        )
        self.lbl_metricas_titulo.grid(row=0, column=0, sticky="w", padx=18, pady=(18, 10))

        self.lbl_accuracy = ctk.CTkLabel(self.card_metricas, text="Accuracy: —", font=ctk.CTkFont(size=16))
        self.lbl_accuracy.grid(row=1, column=0, sticky="w", padx=18, pady=4)

        self.lbl_auc = ctk.CTkLabel(self.card_metricas, text="AUC: —", font=ctk.CTkFont(size=16))
        self.lbl_auc.grid(row=2, column=0, sticky="w", padx=18, pady=4)

        self.lbl_precision = ctk.CTkLabel(self.card_metricas, text="Precision Mais: —", font=ctk.CTkFont(size=16))
        self.lbl_precision.grid(row=3, column=0, sticky="w", padx=18, pady=4)

        self.lbl_recall = ctk.CTkLabel(self.card_metricas, text="Recall Mais: —", font=ctk.CTkFont(size=16))
        self.lbl_recall.grid(row=4, column=0, sticky="w", padx=18, pady=4)

        self.lbl_f1 = ctk.CTkLabel(self.card_metricas, text="F1 Mais: —", font=ctk.CTkFont(size=16))
        self.lbl_f1.grid(row=5, column=0, sticky="w", padx=18, pady=4)

        self.lbl_baseline = ctk.CTkLabel(self.card_metricas, text="Baseline: —", font=ctk.CTkFont(size=16))
        self.lbl_baseline.grid(row=6, column=0, sticky="w", padx=18, pady=4)

        self.lbl_ganho = ctk.CTkLabel(self.card_metricas, text="Ganho vs baseline: —", font=ctk.CTkFont(size=16))
        self.lbl_ganho.grid(row=7, column=0, sticky="w", padx=18, pady=(4, 18))

    def executar_previsao(self):
        try:
            if not self.catalogo:
                raise ValueError("Nenhum modelo foi encontrado na pasta configurada.")

            rodovia = self.combo_rodovia.get().strip()
            meta = self.entry_meta.get().strip().replace(",", ".")
            threshold = self.entry_threshold.get().strip().replace(",", ".")

            if not rodovia:
                raise ValueError("Informe a rodovia.")

            if not meta:
                raise ValueError("Informe a meta de referência.")

            meta = float(meta)
            threshold = float(threshold) if threshold else 0.60

            if threshold < 0 or threshold > 1:
                raise ValueError("O threshold deve estar entre 0 e 1.")

            resultado = prever_aposta(
                catalogo=self.catalogo,
                rodovia_identificacao=rodovia,
                meta_referencia=meta,
                threshold_confianca=threshold
            )

            self.mostrar_resultado(resultado)

        except Exception as e:
            self.lbl_status.configure(text=f"Erro: {str(e)}", text_color="#ff6b6b")

    def mostrar_resultado(self, resultado):
        confianca_pct = resultado["confianca"] * 100
        prob_mais_pct = resultado["prob_mais"] * 100
        prob_ate_pct = resultado["prob_ate"] * 100
        threshold_pct = resultado["threshold_utilizado"] * 100

        cor_aposta = "#2fbf71" if resultado["apostar"] else "#ffb703"
        texto_aposta = "APOSTAR" if resultado["apostar"] else "NÃO APOSTAR"

        self.lbl_status.configure(text="Previsão realizada com sucesso.", text_color="#7dd3fc")
        cor_previsao = "#2fbf71" if resultado["previsao"].startswith("Mais de") else "#ff4d4f"
        self.lbl_previsao.configure(text=resultado["previsao"], text_color=cor_previsao)
        self.lbl_apostar.configure(
            text=f"Decisão: {texto_aposta}",
            text_color=cor_aposta
        )
        self.lbl_confianca.configure(
            text=f"Confiança do modelo: {confianca_pct:.2f}% | Threshold: {threshold_pct:.2f}%"
        )
        self.lbl_prob_mais.configure(text=f"Probabilidade de Mais de {resultado['meta_referencia']:.0f}: {prob_mais_pct:.2f}%")
        self.lbl_prob_ate.configure(text=f"Probabilidade de Até {resultado['meta_referencia']:.0f}: {prob_ate_pct:.2f}%")
        self.lbl_horario.configure(text=f"Horário usado: {resultado['horario_execucao']}")
        self.lbl_rodovia.configure(text=f"Rodovia: {resultado['rodovia']}")
        self.lbl_meta.configure(text=f"Meta de referência: {resultado['meta_referencia']:.0f}")

        metricas = resultado["metricas_modelo"]

        self.lbl_accuracy.configure(text=f"Accuracy: {self._fmt_pct(metricas.get('accuracy'))}")
        self.lbl_auc.configure(text=f"AUC: {self._fmt_pct(metricas.get('auc'))}")
        self.lbl_precision.configure(text=f"Precision Mais: {self._fmt_pct(metricas.get('precision_mais'))}")
        self.lbl_recall.configure(text=f"Recall Mais: {self._fmt_pct(metricas.get('recall_mais'))}")
        self.lbl_f1.configure(text=f"F1 Mais: {self._fmt_pct(metricas.get('f1_mais'))}")
        self.lbl_baseline.configure(text=f"Baseline: {self._fmt_pct(metricas.get('baseline_majoritaria'))}")
        self.lbl_ganho.configure(text=f"Ganho vs baseline: {self._fmt_pct(metricas.get('ganho_vs_baseline'))}")

    def _fmt_pct(self, valor):
        if valor is None:
            return "—"
        return f"{valor * 100:.2f}%"


if __name__ == "__main__":
    app = App()
    app.mainloop()