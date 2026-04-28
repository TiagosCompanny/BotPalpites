import pandas as pd
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

arquivo = Path(r"C:\Users\Tiago Carvalho\Desktop\Projeto Palpites\BotPalpites\Rodovias\DadosRodovias\passagens_carros_por_mercado.xlsx")

coluna_data = "DataHora"
tamanho_bloco = 100_000

fuso_origem = ZoneInfo("America/New_York")
fuso_destino = ZoneInfo("America/Sao_Paulo")

print("Iniciando correção de fuso horário...")

backup = arquivo.with_name(
    f"{arquivo.stem}_backup_antes_corrigir_fuso_{datetime.now().strftime('%Y%m%d_%H%M%S')}{arquivo.suffix}"
)

print("Lendo planilha...")
df = pd.read_excel(arquivo)

total_linhas = len(df)
print(f"Total de linhas encontradas: {total_linhas}")

if coluna_data not in df.columns:
    raise ValueError(f"Coluna '{coluna_data}' não encontrada na planilha.")

print("Criando backup...")
df.to_excel(backup, index=False)
print(f"Backup criado em: {backup}")

print("Convertendo coluna DataHora para datetime...")
df[coluna_data] = pd.to_datetime(df[coluna_data], errors="coerce")

datas_invalidas = df[coluna_data].isna().sum()
print(f"Datas inválidas encontradas: {datas_invalidas}")

print(f"Fuso origem: {fuso_origem}")
print(f"Fuso destino: {fuso_destino}")
print("Corrigindo fuso horário em blocos...")

for inicio in range(0, total_linhas, tamanho_bloco):
    fim = min(inicio + tamanho_bloco, total_linhas)

    bloco = df.loc[inicio:fim - 1, coluna_data]

    df.loc[inicio:fim - 1, coluna_data] = (
        bloco
        .dt.tz_localize(fuso_origem, ambiguous="NaT", nonexistent="shift_forward")
        .dt.tz_convert(fuso_destino)
        .dt.tz_localize(None)
    )

    percentual = (fim / total_linhas) * 100
    print(f"Progresso: {fim}/{total_linhas} linhas corrigidas ({percentual:.2f}%)")

print("Salvando planilha corrigida...")
df.to_excel(arquivo, index=False)

print("Planilha corrigida com sucesso.")
print(f"Backup criado em: {backup}")
print(f"Arquivo atualizado: {arquivo}")