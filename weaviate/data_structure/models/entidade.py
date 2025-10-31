from .documento import Documento

class Entidade:
    def __init__(self, id, nome, documentos=None):
        self.id = id
        self.nome = nome
        self.documentos = [doc for doc in documentos if isinstance(doc, Documento)] if documentos else []

    def update(self, id=None, nome=None, documentos=None):
        if id: self.id = id
        if nome: self.nome = nome
        if documentos is not None: self.documentos = [doc for doc in documentos if isinstance(doc, Documento)]

    def add_documento(self, documento: Documento):
        if isinstance(documento, Documento):
            self.documentos.append(documento)

    def remove_documento(self, documento: Documento):
        if documento in self.documentos:
            self.documentos.remove(documento)

    def __repr__(self):
        return f"Entidade(id='{self.id}', nome='{self.nome}', documentos={self.documentos})"

    def __str__(self):
        return f"Entidade('{self.nome}', documentos={len(self.documentos)})"