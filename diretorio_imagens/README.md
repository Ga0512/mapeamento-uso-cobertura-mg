# Estrutura de Diretórios – Produto P4 (Classificação)

Este repositório organiza os resultados do **Produto P4 – Classificação de Imagens** do projeto **Mapeamento de Cobertura Vegetal e Uso da Terra no Estado de Minas Gerais**.  

A estrutura segue a **regionalização por sub-bacias hidrográficas**, conforme definido no Plano de Trabalho.  
Cada sub-bacia possui a mesma estrutura de pastas internas, garantindo **padronização, rastreabilidade e reprodutibilidade** dos resultados.

📂 **Acesso ao diretório completo:**  
[Link para SharePoint](https://ge21cm.sharepoint.com/:f:/s/SAMARCO/El95v9eODuFDnG949-1_yVsBdzoLgiH6uomiwfDPym9VHg?e=cbLddx)

---

## Estrutura em Árvore

```bash
Planet
├── DO1
│   ├── Amostras
│   ├── Classificacao
│   ├── Dados_auxiliares
│   ├── Equalizado
│   ├── Final
│   ├── Poligonos
│   └── Mosaic_Planet_Regional1_DO1_final.tif
│
├── DO2
│   ├── Amostras
│   ├── Classificacao
│   ├── Dados_auxiliares
│   ├── Equalizado
│   ├── Final
│   ├── Poligonos
│   └── Mosaic_Planet_Regional1_DO2_final.tif
│
├── GD2
│   ├── Amostras
│   ├── Classificacao
│   ├── Dados_auxiliares
│   ├── Equalizado
│   ├── Final
│   ├── Poligonos
│   └── Mosaic_Planet_Regional1_GD2_final.tif
│
├── SF1
│   ├── Amostras
│   ├── Classificacao
│   ├── Dados_auxiliares
│   ├── Equalizado
│   ├── Final
│   ├── Poligonos
│   └── Mosaic_Planet_Regional1_SF1_final.tif
│
├── SF2
│   ├── Amostras
│   ├── Classificacao
│   ├── Dados_auxiliares
│   ├── Equalizado
│   ├── Final
│   ├── Poligonos
│   └── Mosaic_Planet_Regional1_SF2_final.tif
│
├── SF3
│   ├── Amostras
│   ├── Classificacao
│   ├── Dados_auxiliares
│   ├── Equalizado
│   ├── Final
│   ├── Poligonos
│   └── Mosaic_Planet_Regional1_SF3_final.tif
│
├── SF5
│   ├── Amostras
│   ├── Classificacao
│   ├── Dados_auxiliares
│   ├── Equalizado
│   ├── Final
│   ├── Poligonos
│   └── Mosaic_Planet_Regional1_SF5_final.tif

```

Descrição das Pastas e Arquivos 

**Amostras/**
- Contém o conjunto total de pontos amostrais georreferenciados e rotulados, específicos para cada sub-bacia.

- Definidos manualmente a partir da análise de imagens de sensoriamento.

- Complementam os pontos de campo, enriquecendo a base amostral.

- Garantem maior precisão na definição das classes de uso e cobertura da terra.

**Classificacao/**
- Armazena os resultados da classificação do uso da terra e cobertura vegetal.

- Versões brutas e filtradas (com limpeza de pixels ruidosos).

- Logs de execução com estatísticas de treinamento, validação e predição.

- Matrizes de confusão e de validação.

**Dados_auxiliares/**
- Inclui todas as camadas auxiliares utilizadas na classificação final:

- Modelos derivados: DEM (elevação), CHM (altura do dossel).

- Classificação preliminar: class_2 (baseada em Sentinel-2B).

- Texturas: Textura_pca1, Textura_pca2 (estatísticas de matriz de co-ocorrência).

- Embeddings: representações vetoriais extraídas para auxiliar no aprendizado.

**Equalizado/**
- Contém as imagens Planet equalizadas que compõem a área da sub-bacia.

- Equalização garante padronização radiométrica para posterior processamento.

**Final/**
- Armazena as imagens equalizadas acrescidas (empilhadas) com os dados auxiliares.

- Resultado intermediário pronto para a etapa de classificação.

- Mosaic_Planet_Regional*_final.tif
Arquivo raster final da sub-bacia.

- Mosaico composto por 11 bandas, incluindo espectrais, derivadas e texturais.

- Utilizado diretamente como entrada para os algoritmos de classificação.

Define parâmetros e metadados relacionados ao processamento de texturas.

Resumo
Cada sub-bacia (ex.: DO1, GD2, SF2) possui uma estrutura padrão e completa:
Amostras, Classificacao, Dados_auxiliares, Equalizado, Final, além do raster final .tif.

Exceção: SF1, que possui apenas Dados_auxiliares e Equalized.

Estrutura garante padronização, rastreabilidade e reprodutibilidade dos resultados.



