import argparse
import os
import yaml
from src.train.train import segformer, deeplab, unet

def load_config(config_path: str):
    """Carrega configurações do arquivo YAML."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Treinamento de modelos de segmentação com MLflow")
    parser.add_argument("--config", type=str, help="Arquivo YAML com as configurações de treino")
    parser.add_argument("--model", type=str, choices=["segformer", "deeplab", "unet"],
                        help="(Opcional) Sobrescreve o modelo definido no YAML")
    args = parser.parse_args()

    if not args.config:
        raise ValueError("Por favor, especifique um arquivo de configuração com --config caminho/params.yaml")

    # 🔹 Lê o arquivo YAML
    config = load_config(args.config)

    # Substitui modelo se foi passado via CLI
    if args.model:
        config["model_type"] = args.model
    elif "model" in config:  # compatibilidade
        config["model_type"] = config.pop("model")

    os.makedirs(config["output_dir"], exist_ok=True)

    # Cria o modelo
    if config["model_type"] == "segformer":
        model = segformer(use_weights=True, **config)
    elif config["model_type"] == "deeplab":
        model = deeplab(use_weights=True, **config)
    elif config["model_type"] == "unet":
        model = unet(use_weights=True, **config)
    else:
        raise ValueError("Modelo inválido em config.yaml: use segformer, deeplab ou unet")

    model.train()


if __name__ == "__main__":
    main()
