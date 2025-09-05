# Estrutura de Diretórios - Planet

Este repositório contém a base de dados organizada em subdiretórios relacionados ao projeto **Planet**.  
A estrutura segue um padrão com algumas variações entre as pastas.

---

## Estrutura em Árvore

```bash
Planet
├── DO1
│   ├── Amostras
│   ├── Classification
│   ├── Dados_auxiliares
│   ├── Equalized
│   ├── Final
│   └── Mosaic_Planet_Regional1_DO1_final.tif
│
├── DO2
│   ├── Amostras
│   ├── Classification
│   ├── Dados_auxiliares
│   ├── Equalized
│   ├── Final
│   └── Mosaic_Planet_Regional1_DO2_final.tif
│
├── GD2
│   ├── Amostras
│   ├── Classification
│   ├── Dados_auxiliares
│   ├── Equalized
│   ├── Final
│   └── Mosaic_Planet_Regional1_GD2_final.tif
│
├── SF1
│   ├── Dados_auxiliares
│   └── Equalized
│
├── SF2
│   ├── Amostras
│   ├── Classification
│   ├── Dados_auxiliares
│   ├── Equalized
│   ├── Final
│   └── Mosaic_Planet_Regional1_SF2_final.tif
│
├── SF3
│   ├── Amostras
│   ├── Classification
│   ├── Dados_auxiliares
│   ├── Equalized
│   ├── Final
│   └── Mosaic_Planet_Regional1_SF3_final.tif
│
├── SF5
│   ├── Amostras
│   ├── Classification
│   ├── Dados_auxiliares
│   ├── Equalized
│   ├── Final
│   └── Mosaic_Planet_Regional1_SF5_final.tif
│
└── texture_ready.json
```

Descrição dos Diretórios
Diretórios Completos (DO*, GD*, SF2+)
Contêm a seguinte estrutura:

- Amostras/ → Conjunto de amostras para análise.

- Classification/ → Resultados das classificações.

- Dados_auxiliares/ → Dados auxiliares para processamento.

- Equalized/ → Versões equalizadas dos dados.

- Final/ → Resultados finais processados.

- Mosaic_Planet_Regional1_*_final.tif → Arquivo raster final por região.

Diretório Reduzido (SF1)
Contém apenas:

- Dados_auxiliares/

- Equalized/

Arquivo Extra
- texture_ready.json → Arquivo auxiliar em formato JSON, presente na raiz.

