from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, Optional, Type


class TipoLog(str, Enum):
    PREVISAO = "Previsao"
    ORDEM = "Ordem"
    RESULTADO = "Resultado"
    EXECUCAO = "Execucao"
    TREINAMENTO = "Treinamento"


@dataclass(slots=True)
class PrevisaoLog:
    data_hora: str
    rodovia: str
    market_id: str
    selection_id: Optional[str]
    nome_modelo: str
    classe_prevista: str
    confianca: float
    threshold: float
    meta_referencia: str
    odd_minima_aceita: float


@dataclass(slots=True)
class OrdemLog:
    data_hora: str
    rodovia: str
    market_id: str
    selection_id: str
    tipo_ordem: str
    direcao_aposta: str
    stake: float
    odd_solicitada: float
    ordem_enviada: bool
    ordem_executada: bool
    status_ordem: str
    order_id: str


@dataclass(slots=True)
class ResultadoLog:
    data_hora: str
    rodovia: str
    market_id: str
    selection_id: str
    classe_prevista: str
    confianca: float
    aposta_realizada: bool
    direcao_aposta: str
    valor_observado_depois: float
    meta_ou_linha_mercado: str
    acertou_previsao: bool
    aposta_ganha: bool
    lucro_prejuizo: float


@dataclass(slots=True)
class ExecucaoLog:
    data_hora: str
    rodovia: str
    etapa: str
    status: str
    mensagem: str
    market_id: str
    nome_metodo: str
    tempo_execucao_ms: int


@dataclass(slots=True)
class TreinamentoLog:
    data_hora: str
    rodovia: str
    nome_modelo: str
    versao_modelo: str
    algoritmo: str
    arquivo_base: str
    quantidade_registros: int
    quantidade_treino: int
    quantidade_teste: int
    campos_utilizados: str
    campo_alvo: str
    balanceamento_aplicado: str
    acuracia: float
    macro_f1: float
    f1_mais: float
    f1_ate: float
    baseline: float


LOG_MODEL_POR_TIPO: Dict[TipoLog, Type] = {
    TipoLog.PREVISAO: PrevisaoLog,
    TipoLog.ORDEM: OrdemLog,
    TipoLog.RESULTADO: ResultadoLog,
    TipoLog.EXECUCAO: ExecucaoLog,
    TipoLog.TREINAMENTO: TreinamentoLog,
}


class LogService:
    """Serviço central para gravação de logs em CSV por tipo e por dia."""

    def __init__(self, pasta_base: Optional[str] = None) -> None:
        base_path = Path(__file__).resolve().parent
        self.pasta_base = Path(pasta_base) if pasta_base else base_path / "LogApostasRodovias"

    def registrar(self, tipo_log: TipoLog, log: object) -> Path:
        self._validar_tipo_modelo(tipo_log, log)

        pasta_tipo = self.pasta_base / tipo_log.value
        pasta_tipo.mkdir(parents=True, exist_ok=True)

        nome_arquivo = datetime.now().strftime("%Y-%m-%d") + ".csv"
        arquivo_csv = pasta_tipo / nome_arquivo

        colunas = self._colunas_modelo(log)
        registro = self._normalizar_registro(log, colunas)

        arquivo_novo = not arquivo_csv.exists()

        with arquivo_csv.open("a", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=colunas, delimiter=";")
            if arquivo_novo:
                writer.writeheader()
            writer.writerow(registro)

        return arquivo_csv

    def registrar_previsao(self, log: PrevisaoLog) -> Path:
        return self.registrar(TipoLog.PREVISAO, log)

    def registrar_ordem(self, log: OrdemLog) -> Path:
        return self.registrar(TipoLog.ORDEM, log)

    def registrar_resultado(self, log: ResultadoLog) -> Path:
        return self.registrar(TipoLog.RESULTADO, log)

    def registrar_execucao(self, log: ExecucaoLog) -> Path:
        return self.registrar(TipoLog.EXECUCAO, log)

    def registrar_treinamento(self, log: TreinamentoLog) -> Path:
        return self.registrar(TipoLog.TREINAMENTO, log)

    @staticmethod
    def agora_str() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _validar_tipo_modelo(self, tipo_log: TipoLog, log: object) -> None:
        modelo_esperado = LOG_MODEL_POR_TIPO[tipo_log]
        if not isinstance(log, modelo_esperado):
            raise TypeError(
                f"Tipo de objeto inválido para {tipo_log.value}. "
                f"Esperado: {modelo_esperado.__name__}. Recebido: {type(log).__name__}"
            )

    @staticmethod
    def _colunas_modelo(log: object) -> Iterable[str]:
        return list(asdict(log).keys())

    @staticmethod
    def _normalizar_registro(log: object, colunas: Iterable[str]) -> Dict[str, object]:
        registro = asdict(log)
        return {coluna: registro.get(coluna) for coluna in colunas}
