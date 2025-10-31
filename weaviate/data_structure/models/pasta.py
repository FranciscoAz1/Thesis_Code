from tkinter import E
from .fluxo import Fluxo
from .etapa import Etapa
class Pasta:
    def __init__(self, id, nome, documento=None, entidade=None):
        self.id = id
        self.nome = nome
        self.documento = documento if documento else []
        self.entidade = entidade if entidade else []
        self.fluxo = Fluxo(id, nome)

    def update(self, id=None, nome=None, documento=None):
        if id: self.id = id
        if nome: self.nome = nome
        if documento is not None: self.documento = documento
    
    def add_documento(self, documento):
        self.documento.append(documento)
    
    def add_entidade(self, entidade):
        self.entidade.append(entidade)
    
    def make_event(self,Etapa:Etapa):
        self.fluxo.add_etapa(Etapa)

    def __repr__(self):
        return f"Pasta(id='{self.id}', nome='{self.nome}', documento={self.documento})"

    def __str__(self):
        return f"Pasta('{self.nome}', {len(self.documento)}, {len(self.entidade)})"
