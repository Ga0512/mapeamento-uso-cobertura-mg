from office365.sharepoint.client_context import ClientContext
from office365.runtime.auth.client_credential import ClientCredential
import os

site_url = "https://ge21cm.sharepoint.com/sites/SAMARCO"
client_id = "5b37cb3a-7e6e-498d-989c-f5e2b0353bb5"
client_secret = "1d9b89ab-85af-4c9e-9701-7f92b4b790c4"

ctx = ClientContext(site_url).with_credentials(ClientCredential(client_id, client_secret))

# Pasta que você quer baixar:
folder_relative_url = "/sites/SAMARCO/Documentos Compartilhados/SAMARCO/08_BASE DE DADOS/Scripts/Amostras_Semantic_seg/8bits"

local_path = r"C:\Downloads\8bits"

# Função para download recursivo:
def download_folder(ctx, sharepoint_folder_url, local_folder):
    folder = ctx.web.get_folder_by_server_relative_url(sharepoint_folder_url)
    ctx.load(folder)
    ctx.execute_query()

    os.makedirs(local_folder, exist_ok=True)

    # bajar arquivos
    files = folder.files
    ctx.load(files)
    ctx.execute_query()
    for f in files:
        name = f.properties["Name"]
        local_file_path = os.path.join(local_folder, name)
        with open(local_file_path, "wb") as local_file:
            f.download(local_file).execute_query()
        print("Downloaded file:", name)

    # bajar subpastas
    subfolders = folder.folders
    ctx.load(subfolders)
    ctx.execute_query()
    for sf in subfolders:
        sub_name = sf.properties["Name"]
        new_sp_url = sharepoint_folder_url + "/" + sub_name
        new_local = os.path.join(local_folder, sub_name)
        download_folder(ctx, new_sp_url, new_local)

# Uso:
download_folder(ctx, folder_relative_url, local_path)
