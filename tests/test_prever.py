import pytest

from prever import carregar_artefato, classificar, main, validar_medidas


@pytest.fixture(scope="module")
def artefato():
    return carregar_artefato()


def test_classifica_setosa(artefato):
    # primeira flor do dataset Iris, que é uma setosa
    especie, _ = classificar([5.1, 3.5, 1.4, 0.2], artefato)
    assert especie == "setosa"


def test_classifica_virginica(artefato):
    especie, _ = classificar([6.3, 3.3, 6.0, 2.5], artefato)
    assert especie == "virginica"


def test_probabilidades_somam_um(artefato):
    _, probabilidades = classificar([5.9, 3.0, 4.2, 1.5], artefato)
    assert set(probabilidades) == {"setosa", "versicolor", "virginica"}
    assert sum(probabilidades.values()) == pytest.approx(1.0)


def test_medida_negativa_gera_erro(artefato):
    with pytest.raises(ValueError, match="maior que zero"):
        validar_medidas([5.1, -3.5, 1.4, 0.2], artefato)


def test_quantidade_errada_de_medidas_gera_erro(artefato):
    with pytest.raises(ValueError, match="4 medidas"):
        validar_medidas([5.1, 3.5, 1.4], artefato)


def test_medida_fora_da_faixa_gera_aviso(artefato):
    # pétala de 9 cm: a maior pétala do dataset tem 6,9 cm
    avisos = validar_medidas([5.1, 3.5, 9.0, 0.2], artefato)
    assert len(avisos) == 1
    assert "comprimento da pétala" in avisos[0]


def test_medidas_normais_nao_geram_aviso(artefato):
    assert validar_medidas([5.1, 3.5, 1.4, 0.2], artefato) == []


def test_main_aceita_virgula(capsys):
    assert main(["5,1", "3,5", "1,4", "0,2"]) == 0
    assert "Espécie identificada: setosa" in capsys.readouterr().out


def test_main_retorna_erro_para_medida_invalida(capsys):
    assert main(["5.1", "0", "1.4", "0.2"]) == 1
    assert "Erro:" in capsys.readouterr().err
