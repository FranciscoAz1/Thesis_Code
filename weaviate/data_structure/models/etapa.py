import re
from .documento import Documento, Ficheiro, Metadados
class Etapa:
    def __init__(self, id, subject, description, document):
        self.id = id
        self.subject = subject
        self.description = description
        self.document = document

    def __repr__(self):
        return f"Etapa(id='{self.id}', nome='{self.subject}', description='{self.description}', document={self.document})"

    def __str__(self):
        return f"Etapa('{self.subject}', '{self.description}')"
