from office365.sharepoint.client_context import ClientContext
from office365.runtime.auth.client_credential import ClientCredential
import os
from dotenv import load_dotenv
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]  

ENV_PATH = ROOT_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH)

site_url = "https://ge21cm.sharepoint.com/sites/SAMARCO"
client_id = ""
client_secret = ""

ctx = ClientContext(site_url).with_credentials(ClientCredential(client_id, client_secret))

lists = ctx.web.lists
ctx.load(lists)
ctx.execute_query()

print("📚 Bibliotecas disponíveis no site SAMARCO:\n")
for l in lists:
    title = l.properties["Title"]
    root_folder = l.properties["RootFolder"]["ServerRelativeUrl"]
    print(f"➡ {title} → {root_folder}")