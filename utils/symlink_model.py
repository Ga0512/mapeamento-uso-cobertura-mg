import os


def create_symlink(input, output):

    src = os.path.abspath(input)   
    dst = os.path.abspath(output)

    
    if not os.path.exists(src):
        raise FileNotFoundError(f"Origem não encontrada: {src}")


    if os.path.islink(dst) or os.path.exists(dst):
        os.remove(dst)


    try:
        os.symlink(src, dst, target_is_directory=True)
        print("✅ Symlink criado com sucesso!")
    except OSError as e:
        print("Erro:", e)
