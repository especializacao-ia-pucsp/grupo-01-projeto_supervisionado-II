# Projeto Supervisionado II

### Especialização em IA
### PUC-SP

## Grupo I

- Anderson Almeida Costa (RA00306181)
- Rafael de Sousa Camelo (RA00375190)
- Roberto Freires Batista (RA00377469)
- Wesley Rodrigues Silva (RA00342561)

## Objetivo

Desenvolver um modelo capaz de classificar automaticamente a categoria dos chamados utilizando múltiplas modalidades de dados.

## Estrutura do Repositorio

```
+- README.md: Este README descrevendo o projeto e o repositório
|
+- LICENSE: Licença do código neste repositório
|
+- data
|   |
|   +- raw: o arquivo chamados.csv e as imagens originais
|   |
|   +- preprocessed: arquivos com representações intermediárias
|   |
|   +- processed: arquivo chamados.csv pronto para ser utilizado pelo 
|                 modelo mlb_emb
+- models
|   |
|   +- mlb_emb.joblib: modelo gerado
|
+- src
|   |
|   +- preprocessing.py: pipeline de pré-processamento
|
+- notebooks
    |
    +- 01_Exploração.ipynb: Exploração inicial dos dados
    |
    +- 02_Engenharia_dados.ipynb: Análise e pré-processamento
    |
    +- 03_Deep_Learning_v1.ipynb: Modelo de Deep Learning com os dados
    |                             tabulares
    |
    +- 04_Visao_Computacional.ipynb: Modelo de classificação dos chamados
    |                                com base nas imagens
    |
    +- 05_PLN_v1.ipynb: Modelo de classificação dos chamados com base nos 
    |                   títulos e descrições
    |
    +- 06_Modelo_Multimodal.ipynb: Modelo de classificação combinado com
    |                              os dados tabulares e as imagens
    |
    +- 07_Pipeline_Final.ipynb: Pipeline completo lendo os dados raw,
                                gerando e escrevendo a representação
                                intermediária, fazendo o processamento
                                para treinar o modelo final, treinando
                                e validando o modelo e, por fim, salvando
                                o modelo para futuro deploy 
```

## Uso de Inteligência Artificial (IA)

Este projeto contou com a assistência de ferramentas de Inteligência Artificial (como o Gemini integrado ao Colab) durante o seu desenvolvimento. 

A IA foi utilizada principalmente para:
* **Refatoração e Otimização:** Melhoria na estrutura e performance do código.
* **Documentação:** Auxílio na escrita e revisão deste arquivo README e comentários.
* **Resolução de Bugs:** Suporte no diagnóstico de erros pontuais.

Todo o código gerado ou sugerido por IA foi revisado, testado e validado manualmente para garantir a segurança, qualidade e conformidade com as boas práticas.

