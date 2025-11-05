import time
import requests
import subprocess

REPO = "gabriel0murilo/ief"
TAG = "latest"
INTERVAL = 60  # segundos

def get_last_digest():
    url = f"https://hub.docker.com/v2/repositories/{REPO}/tags/{TAG}"
    r = requests.get(url)
    r.raise_for_status()
    return r.json()["images"][0]["digest"]

def main():
    last_digest = get_last_digest()
    print(f"Digest inicial: {last_digest}")

    while True:
        time.sleep(INTERVAL)
        try:
            new_digest = get_last_digest()
            if new_digest != last_digest:
                print("Nova imagem detectada!")
                last_digest = new_digest
                # Aqui você pode disparar alguma ação:
                subprocess.run(["kubectl", "rollout", "restart", "deployment", "ief-deployment"])
        except Exception as e:
            print("Erro ao checar:", e)

if __name__ == "__main__":
    main()
