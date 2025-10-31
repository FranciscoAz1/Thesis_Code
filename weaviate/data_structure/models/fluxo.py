from .etapa import Etapa

class Fluxo:
    def __init__(self, id, nome, etapas=None):
        self.id = id
        self.nome = nome
        self.etapas = []

    def update(self, id=None, nome=None, etapas=None):
        if id: self.id = id
        if nome: self.nome = nome
        if etapas is not None: self.etapas = [etapa for etapa in etapas if isinstance(etapa, Etapa)]
    
    def add_etapa(self, etapa):
        self.etapas.append(etapa)

    def __repr__(self):
        return f"Fluxo(id='{self.id}', nome='{self.nome}', etapas={self.etapas})"

    def __str__(self):
        return f"Fluxo('{self.nome}', etapas={len(self.etapas)})"