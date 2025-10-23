import subprocess
import time
import requests

def wait_for_server(url, timeout=15):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url)
            if r.status_code == 200:
                return True
        except requests.exceptions.ConnectionError:
            time.sleep(1)
    raise RuntimeError("Servidor não respondeu dentro do tempo limite.")

def test_server_is_running():
    process = subprocess.Popen(
        ["uvicorn", "pipeline.api.main:app", "--host", "127.0.0.1", "--port", "8000", "--log-level", "debug", "--reload"],
        stdout=subprocess.DEVNULL,  # não bloqueia
        stderr=subprocess.DEVNULL,
    )

    try:
        # espera o servidor responder
        wait_for_server("http://127.0.0.1:8000/health", timeout=30)

        # testa o endpoint
        response = requests.get("http://127.0.0.1:8000/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    finally:
        print("Done ✅")
        process.terminate()
        process.wait()
