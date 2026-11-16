"""Classifica a espécie de uma flor Iris a partir das suas 4 medidas.

Usa o modelo treinado e salvo pelo notebook notebooks/classificacao_iris.ipynb.

Uso:
    python prever.py 6.3 3.3 6.0 2.5

As medidas são, nesta ordem e em centímetros: comprimento da sépala,
largura da sépala, comprimento da pétala e largura da pétala.
"""

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
import sklearn

CAMINHO_MODELO = Path(__file__).parent / "modelo" / "modelo_iris.pkl"

NOMES_MEDIDAS = [
    "comprimento da sépala",
    "largura da sépala",
    "comprimento da pétala",
    "largura da pétala",
]


def carregar_artefato(caminho=CAMINHO_MODELO):
    """Carrega o dicionário salvo pelo notebook (modelo, nomes e limites)."""
    if not caminho.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em {caminho}. "
            "Rode o notebook notebooks/classificacao_iris.ipynb para gerá-lo."
        )
    return joblib.load(caminho)


def validar_medidas(medidas, artefato):
    """Verifica as medidas informadas.

    Levanta ValueError se a entrada for inválida (quantidade errada ou valor
    menor ou igual a zero). Devolve uma lista de avisos para as medidas que
    estão fora da faixa vista no treino: o modelo ainda responde, mas o
    resultado pode não ser confiável.
    """
    if len(medidas) != len(NOMES_MEDIDAS):
        raise ValueError(f"São necessárias {len(NOMES_MEDIDAS)} medidas, mas foram informadas {len(medidas)}.")

    avisos = []
    for nome, coluna, valor in zip(NOMES_MEDIDAS, artefato["nomes_caracteristicas"], medidas):
        if valor <= 0:
            raise ValueError(f"O {nome} deve ser maior que zero (recebido: {valor}).")

        minimo, maximo = artefato["limites"][coluna]
        if not minimo <= valor <= maximo:
            avisos.append(
                f"{nome} = {valor} cm está fora da faixa vista no treino ({minimo} a {maximo} cm)."
            )
    return avisos


def classificar(medidas, artefato):
    """Devolve a espécie prevista e a probabilidade de cada espécie."""
    # mesmo formato usado no treino: DataFrame com os nomes das colunas
    flor = pd.DataFrame([medidas], columns=artefato["nomes_caracteristicas"])
    modelo = artefato["modelo"]

    indice = modelo.predict(flor)[0]
    probabilidades = modelo.predict_proba(flor)[0]

    especie = artefato["nomes_classes"][indice]
    return especie, dict(zip(artefato["nomes_classes"], probabilidades))


def numero(texto):
    """Converte o texto em número, aceitando vírgula como separador decimal (6,3)."""
    try:
        return float(texto.replace(",", "."))
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{texto}' não é um número válido.") from None


def main(argumentos=None):
    parser = argparse.ArgumentParser(
        description="Classifica a espécie de uma flor Iris a partir das 4 medidas (em cm).",
        epilog="Exemplo: python prever.py 6.3 3.3 6.0 2.5",
    )
    parser.add_argument("comprimento_sepala", type=numero, help="comprimento da sépala (cm)")
    parser.add_argument("largura_sepala", type=numero, help="largura da sépala (cm)")
    parser.add_argument("comprimento_petala", type=numero, help="comprimento da pétala (cm)")
    parser.add_argument("largura_petala", type=numero, help="largura da pétala (cm)")
    args = parser.parse_args(argumentos)

    medidas = [args.comprimento_sepala, args.largura_sepala, args.comprimento_petala, args.largura_petala]

    try:
        artefato = carregar_artefato()
        avisos = validar_medidas(medidas, artefato)
    except (FileNotFoundError, ValueError) as erro:
        print(f"Erro: {erro}", file=sys.stderr)
        return 1

    if artefato.get("versao_sklearn") != sklearn.__version__:
        print(
            f"Atenção: o modelo foi salvo com scikit-learn {artefato.get('versao_sklearn')}, "
            f"mas a versão instalada é {sklearn.__version__}. "
            "Se houver erro, rode o notebook de novo para gerar o modelo.\n"
        )

    for aviso in avisos:
        print(f"Atenção: {aviso}")
    if avisos:
        print("O resultado abaixo pode não ser confiável.\n")

    especie, probabilidades = classificar(medidas, artefato)
    print(f"Espécie identificada: {especie}")
    for nome, probabilidade in probabilidades.items():
        print(f"  {nome}: {probabilidade:.0%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
