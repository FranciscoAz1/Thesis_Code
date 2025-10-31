class Ficheiro:
    def __init__(self, id, nome, tipo, tamanho, metadados=None):
        self.id = id
        self.nome = nome
        self.tipo = tipo
        self.tamanho = tamanho

    def update(self, id=None, nome=None, tipo=None, tamanho=None):
        if id: self.id = id
        if nome: self.nome = nome
        if tipo: self.tipo = tipo
        if tamanho is not None: self.tamanho = tamanho

    def __eq__(self, other):
        if isinstance(other, Ficheiro):
            return (self.id == other.id and 
                    self.nome == other.nome and 
                    self.tipo == other.tipo and 
                    self.tamanho == other.tamanho)
        return False

    def __repr__(self):
        return f"Ficheiro(id='{self.id}', nome='{self.nome}', tipo='{self.tipo}', tamanho={self.tamanho})"

    def __str__(self):
        return f"Ficheiro('{self.nome}')"

class Metadados:
    def __init__(self, id, tipo, valor, ficheiros, categoria=["Destacado", "Outros"]):
        self.id = id
        self.tipo = tipo
        self.valor = valor
        self.categoria = categoria
        self.ficheiros = {}
        for ficheiro in ficheiros:
            if self.valid_ficheiro(ficheiro):
                key = list(ficheiro.keys())[0]   # Extract the key
                value = list(ficheiro.values())[0]  # Extract the value
                self.ficheiros[key] = value  # Store in dictionary properly
    
    def valid_ficheiro(self, ficheiro):
        key = list(ficheiro.keys())[0]  # Extract key
        value = list(ficheiro.values())[0]  # Extract value
        
        if key not in self.categoria:
            raise ValueError(f"Invalid category: {key}")  # Now correctly checking a single key

        if isinstance(value, Ficheiro):  # Check value correctly
            return True
        else:
            raise ValueError(f"Invalid ficheiro: {value}")


    def update(self, id=None, tipo=None, valor=None, ficheiro=None, categoria=None):
        if id: self.id = id
        if tipo: self.tipo = tipo
        if valor: self.valor = valor
        if ficheiro is not None: self.ficheiros = ficheiro
        if categoria is not None: self.categoria = categoria
    
    def add_ficheiro(self, ficheiro:Ficheiro, categoria): 
        if categoria in self.categoria:
            self.ficheiros[categoria].append(ficheiro)
        else:
            raise ValueError(f"Invalid category: {categoria}")

    def __repr__(self):
        return f"Metadados(id='{self.id}', tipo='{self.tipo}', valor='{self.valor}', categoria={self.categoria}, ficheiros={self.ficheiros})"

    def __str__(self):
        return f"Metadados('{self.tipo}', '{self.valor}', {self.ficheiros})"

class Documento:
    def __init__(self, id, nome, ficheiro:Ficheiro, metadados:Metadados=None): 
        self.id = id
        self.nome = nome
        self.ficheiro = ficheiro if isinstance(ficheiro, Ficheiro) else None
        self.metadados = [meta for meta in metadados if isinstance(meta, Metadados)] if metadados else []

    def update(self, id=None, nome=None, ficheiro=None, metadados=None):
        if id: self.id = id
        if nome: self.nome = nome
        if ficheiro and isinstance(ficheiro, Ficheiro): self.ficheiro = ficheiro
        if metadados is not None: self.metadados = [meta for meta in metadados if isinstance(meta, Metadados)]
    
    def add_metadados(self, metadados:Metadados):
        # not optimized for a lot of files
        if isinstance(metadados, Metadados):
            if self.ficheiro in metadados.ficheiros.values():
                self.metadados.append(metadados)
            else:
                raise ValueError(f"Invalid Metadados, ficheiro not in Documento: {metadados}")

    def __repr__(self):
        return f"Documento(id={self.id}, nome={self.nome}, ficheiro={self.ficheiro}, metadados={self.metadados})"

    def __str__(self):
        return f"Documento('{self.nome}', {self.ficheiro}, metadados={len(self.metadados)})"
