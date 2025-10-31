import sys
import os
from tkinter import E

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models import Pasta, Entidade, Fluxo, Etapa, Metadados, Ficheiro, Documento

def main():
    # Create example files
    file1 = Ficheiro("1", "document.txt", "txt", 1200)
    file2 = Ficheiro("2", "image.png", "png", 1200)

    # Create metadata
    meta1 = Metadados("1", "Author", "Francisco", [{"Destacado": file1}, {"Outros": file2}])

    # Create a documento with the file, metadata
    documento1 = Documento("1", "Documento 1", file1, metadados=[meta1])

    # Create a entidade
    entidade1 = Entidade("1", "IST")
    entidade1.add_documento(documento1)

    # Create a folder
    pasta1 = Pasta("1", "Projetos")
    pasta1.add_documento(documento1)
    pasta1.add_entidade(entidade1)

    # Create a etapa
    etapa1 = Etapa("1", "Student", "enrollment", documento1)
    pasta1.make_event(etapa1)

    # Print initial state
    print("Initial state:")
    print(f"File: {file1}")
    print(f"Metadata: {meta1}")
    print(f"Documento: {documento1}")
    print(f"Entidade: {entidade1}")
    print(f"Pasta: {pasta1}")
    print(f"Etapa: {etapa1}")
    print(f"Fluxo in Pasta: {pasta1.fluxo}")

    # Add another etapa
    etapa2 = Etapa("2", "IST", "acceptance", documento1)
    pasta1.make_event(etapa2)

    # Add more metadata to the documento
    meta2 = Metadados("2", "Update", "Student accepted", [{"Destacado": file1}, {"Outros": file2}])
    documento1.add_metadados(meta2)

    # Print updated state
    print("\nUpdated state:")
    print(f"File: {file1}")
    print(f"Metadata: {meta1}")
    print(f"Additional Metadata: {meta2}")
    print(f"Documento: {documento1}")
    print(f"Entidade: {entidade1}")
    print(f"Pasta: {pasta1}")
    print(f"Etapa 1: {etapa1}")
    print(f"Etapa 2: {etapa2}")
    print(f"Fluxo in Pasta: {pasta1.fluxo}")

if __name__ == "__main__":
    main()
