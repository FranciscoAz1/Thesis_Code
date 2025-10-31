import weaviate
import time
from weaviate.classes.config import Property, DataType, ReferenceProperty, Configure
from weaviate.classes.query import QueryReference, MetadataQuery
from typing import List, Dict, Any

class WeaviateDataManager:
    # Define Ollama API details
    OLLAMA_API_ENDPOINT = "http://host.docker.internal:11434"
    VECTOR_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    GENERATION_MODEL = "qwen2.5:latest"
    
    def __init__(self, connect_to_local=True):
        """Initialize the Weaviate client connection."""
        if connect_to_local:
            self.client = weaviate.connect_to_local()
        else:
            # Add configuration for non-local connection if needed
            self.client = weaviate.connect_to_local()  # Placeholder
        
        # Collection objects
        self.fluxo = None
        self.etapa = None
        self.entidade = None
        self.pasta = None
        self.ficheiro = None
        self.metadados = None
    
    def setup_collections(self, clean_start=True):
        """Create all collections and their references."""
        # Delete old collections if clean_start is True
        if clean_start:
            self._delete_existing_collections()
        
        # Create collections
        self._create_collections()
        
        # Get collection objects
        self._get_collection_objects()
        
        # Create references between collections
        self._create_references()
        
        print("Collections with vectorization & generation created successfully!")
        print("References added successfully!")
    
    def _delete_existing_collections(self):
        """Delete existing collections if they exist."""
        collection_names = ["Fluxo", "Etapa", "Entidade", "Pasta", "Ficheiro", "Metadados"]
        for collection_name in collection_names:
            try:
                self.client.collections.delete(collection_name)
            except:
                pass  # Ignore if collection doesn't exist
    
    def _create_collections(self):
        """Create all required collections."""
        # 1. Create "Fluxo" Collection
        self._create_collection(
            "Fluxo",
            "A workflow that contains multiple stages (etapas).",
            [Property(name="name", data_type=DataType.TEXT)]
        )

        # 2. Create "Etapa" Collection
        self._create_collection(
            "Etapa",
            "A stage within a workflow.",
            [Property(name="name", data_type=DataType.TEXT)]
        )

        # 3. Create "Entidade" Collection
        self._create_collection(
            "Entidade",
            "An entity that owns folders (pastas).",
            [Property(name="name", data_type=DataType.TEXT)]
        )

        # 4. Create "Pasta" Collection
        self._create_collection(
            "Pasta",
            "A folder belonging to an entity and containing documents (ficheiros).",
            [Property(name="name", data_type=DataType.TEXT)]
        )

        # 5. Create "Ficheiro" Collection
        self._create_collection(
            "Ficheiro",
            "A document that contains metadata.",
            [Property(name="name", data_type=DataType.TEXT),
             Property(name="metadados", data_type=DataType.TEXT)]
        )

        # # 6. Create "Metadados" Collection
        # self._create_collection(
        #     "Metadados",
        #     "Metadata associated with a document (ficheiro).",
        #     [Property(name="name", data_type=DataType.TEXT)]
        # )
    
    def _create_collection(self, name, description, properties):
        """Create a collection with vectorization & generative AI."""
        self.client.collections.create(
            name=name,
            description=description,
            properties=properties,
        vectorizer_config=[
            Configure.NamedVectors.text2vec_transformers(
                name="text_vector",
                source_properties=["text"],
                pooling_strategy="masked_mean",
                )
        ],
            generative_config=Configure.Generative.ollama(
                api_endpoint=self.OLLAMA_API_ENDPOINT,
                model=self.GENERATION_MODEL
            )
        )
    
    def _get_collection_objects(self):
        """Get references to all collection objects."""
        self.fluxo = self.client.collections.get("Fluxo")
        self.etapa = self.client.collections.get("Etapa")
        self.entidade = self.client.collections.get("Entidade")
        self.pasta = self.client.collections.get("Pasta")
        self.ficheiro = self.client.collections.get("Ficheiro")
        # self.metadados = self.client.collections.get("Metadados")
    
    def _create_references(self):
        """Create all references between collections."""
        # Add cross-references
        self.fluxo.config.add_reference(ReferenceProperty(name="hasEtapas", target_collection="Etapa"))
        self.fluxo.config.add_reference(ReferenceProperty(name="belongsToFicheiros", target_collection="Ficheiro"))
        self.fluxo.config.add_reference(ReferenceProperty(name="belongsToPastas", target_collection="Pasta"))

        self.etapa.config.add_reference(ReferenceProperty(name="belongsToFluxo", target_collection="Fluxo"))
        self.etapa.config.add_reference(ReferenceProperty(name="hasFicheiros", target_collection="Ficheiro"))

        self.entidade.config.add_reference(ReferenceProperty(name="hasFicheiros", target_collection="Ficheiro"))
        self.entidade.config.add_reference(ReferenceProperty(name="hasPastas", target_collection="Pasta"))

        self.pasta.config.add_reference(ReferenceProperty(name="hasFicheiros", target_collection="Ficheiro"))
        self.pasta.config.add_reference(ReferenceProperty(name="hasEntidades", target_collection="Entidade"))
        self.pasta.config.add_reference(ReferenceProperty(name="hasFluxos", target_collection="Fluxo"))

        # self.ficheiro.config.add_reference(ReferenceProperty(name="belongsToMetadados", target_collection="Metadados"))
        self.ficheiro.config.add_reference(ReferenceProperty(name="hasEtapas", target_collection="Etapa"))
        self.ficheiro.config.add_reference(ReferenceProperty(name="hasPastas", target_collection="Pasta"))
        self.ficheiro.config.add_reference(ReferenceProperty(name="hasEntidades", target_collection="Entidade"))

        # self.metadados.config.add_reference(ReferenceProperty(name="hasFicheiros", target_collection="Ficheiro"))
        # self.metadados.config.add_reference(ReferenceProperty(name="hasEtapas", target_collection="Etapa"))
        # self.metadados.config.add_reference(ReferenceProperty(name="hasPastas", target_collection="Pasta"))
        # self.metadados.config.add_reference(ReferenceProperty(name="hasEntidades", target_collection="Entidade"))
    
    def insert_sample_data(self, num_entidades=2, num_pastas_per_entidade=5, 
                           num_ficheiros_per_pasta=8, num_metadados_per_ficheiro=1,
                           num_fluxos=5, num_etapas_per_fluxo=10):
        """Insert sample data into all collections with proper references."""
        # Step 1: Create Entidades
        entidade_objs = [self.add_entidade(f"Empresa {i+1}") for i in range(num_entidades)]

        # Step 2: Create Pastas and link them to Entidades
        pasta_objs = []
        for i, entidade_obj in enumerate(entidade_objs):
            for j in range(num_pastas_per_entidade):
                pasta_obj = self.add_pasta(f"Pasta {j+1} of Empresa {i+1}", entidade_obj=entidade_obj)
                pasta_objs.append(pasta_obj)

        # Step 3: Create Ficheiros and link them to Pastas and Entidades
        ficheiro_objs = []
        for i, pasta_obj in enumerate(pasta_objs):
            entidade_obj = entidade_objs[i % len(entidade_objs)]
            for j in range(num_ficheiros_per_pasta):
                ficheiro_obj = self.add_ficheiro(f"Documento {j+1} in {pasta_obj}",
                                                 f"Metadata {j+1}", 
                                                pasta_obj=pasta_obj, 
                                                entidade_obj=entidade_obj)
                ficheiro_objs.append(ficheiro_obj)
        # Step 5: Create Fluxos and link them to Pastas and Ficheiros
        fluxo_objs = []
        for i in range(num_fluxos):
            pasta_obj = pasta_objs[i % len(pasta_objs)]
            ficheiro_obj = ficheiro_objs[i % len(ficheiro_objs)]
            fluxo_obj = self.add_fluxo(f"Fluxo {i+1}", pasta_obj=pasta_obj, ficheiro_obj=ficheiro_obj)
            fluxo_objs.append(fluxo_obj)

        # Step 6: Create Etapas and link them to Fluxos and Ficheiros
        etapa_objs = []
        for i, fluxo_obj in enumerate(fluxo_objs):
            ficheiro_obj = ficheiro_objs[i % len(ficheiro_objs)]
            for j in range(num_etapas_per_fluxo):
                etapa_obj = self.add_etapa(f"Etapa {j+1} of Fluxo {i+1}", 
                                          fluxo_obj=fluxo_obj, 
                                          ficheiro_obj=ficheiro_obj)
                etapa_objs.append(etapa_obj)

        print("Bulk sample data inserted and linked successfully!")
        
        return {
            "entidades": entidade_objs,
            "pastas": pasta_objs,
            "ficheiros": ficheiro_objs,
            # "metadados": metadados_objs,
            "fluxos": fluxo_objs,
            "etapas": etapa_objs
        }

    def insert_conection_data(self, num_entidades: int = 2,
                               num_pastas_per_entidade: int = 5,
                               num_ficheiros_per_pasta: int = 8,
                               num_metadados_per_ficheiro: int = 1,
                               num_fluxos: int = 5,
                               num_etapas_per_fluxo: int = 10):
        """
        Insert data with a denser web of connections, aiming to pass the maximum
        available references into each add_* call and then add symmetrical links.

        Notes:
        - Metadados are modeled as a TEXT property on Ficheiro in this schema; we don't
          create a separate Metadados collection here.
        - We distribute fluxos roughly evenly across all pastas and link each fluxo to
          a Pasta and a Ficheiro at creation time.
        - Each Etapa will link to its Fluxo and one Ficheiro.
        - After creation, we add extra reverse references to maximize connectivity:
          Pasta.hasFluxos, Ficheiro.hasPastas, Ficheiro.hasEntidades, Ficheiro.hasEtapas.
        """

        # 1) Create Entidades and Pastas
        entidade_objs = [self.add_entidade(f"Empresa {i+1}") for i in range(num_entidades)]

        pasta_objs_by_entidade = []  # List[List[pasta]] aligned with entidade index
        all_pastas = []
        for i, entidade_obj in enumerate(entidade_objs):
            pastas_for_entidade = []
            for j in range(num_pastas_per_entidade):
                pasta_obj = self.add_pasta(f"Pasta {j+1} of Empresa {i+1}", entidade_obj=entidade_obj)
                pastas_for_entidade.append(pasta_obj)
                all_pastas.append((entidade_obj, pasta_obj))
            pasta_objs_by_entidade.append(pastas_for_entidade)

        # 2) Create Ficheiros for each Pasta (link to Pasta + Entidade)
        ficheiros_by_pasta = {}  # pasta_obj -> List[ficheiro]
        all_ficheiros = []
        for entidade_obj, pasta_obj in all_pastas:
            ficheiros = []
            for k in range(num_ficheiros_per_pasta):
                ficheiro_obj = self.add_ficheiro(
                    ficheiro_name=f"Documento {k+1} in {pasta_obj}",
                    metadados_name=f"Metadata {k+1}",
                    pasta_obj=pasta_obj,
                    entidade_obj=entidade_obj,
                )
                ficheiros.append(ficheiro_obj)
                all_ficheiros.append((entidade_obj, pasta_obj, ficheiro_obj))
            ficheiros_by_pasta[pasta_obj] = ficheiros

        # 3) Create Fluxos and link them to a Pasta and one Ficheiro (use both args)
        fluxo_objs = []
        total_pastas = len(all_pastas)
        for i in range(num_fluxos):
            entidade_obj, pasta_obj = all_pastas[i % total_pastas]
            pasta_ficheiros = ficheiros_by_pasta[pasta_obj]
            primary_fich = pasta_ficheiros[i % len(pasta_ficheiros)]
            fluxo_obj = self.add_fluxo(
                fluxo_name=f"Fluxo {i+1}",
                pasta_obj=pasta_obj,
                ficheiro_obj=primary_fich,
            )
            fluxo_objs.append((entidade_obj, pasta_obj, fluxo_obj))

        # 4) Create Etapas for each Fluxo and link to Fluxo + one Ficheiro (use both args)
        etapa_objs = []
        for i, (entidade_obj, pasta_obj, fluxo_obj) in enumerate(fluxo_objs):
            pasta_ficheiros = ficheiros_by_pasta[pasta_obj]
            for j in range(num_etapas_per_fluxo):
                fich_for_etapa = pasta_ficheiros[(i + j) % len(pasta_ficheiros)]
                etapa_obj = self.add_etapa(
                    etapa_name=f"Etapa {j+1} of Fluxo {i+1}",
                    fluxo_obj=fluxo_obj,
                    ficheiro_obj=fich_for_etapa,
                )
                etapa_objs.append((pasta_obj, fluxo_obj, etapa_obj, fich_for_etapa))

        # 5) Add extra reverse and cross links for denser connectivity
        # 5a) Pasta.hasFluxos for every fluxo in that pasta
        for entidade_obj, pasta_obj, fluxo_obj in fluxo_objs:
            try:
                self.pasta.data.reference_add(pasta_obj, "hasFluxos", fluxo_obj)
            except Exception:
                pass

        # 5b) Ficheiro.hasPastas and Ficheiro.hasEntidades symmetrical links
        for entidade_obj, pasta_obj, ficheiro_obj in all_ficheiros:
            try:
                self.ficheiro.data.reference_add(ficheiro_obj, "hasPastas", pasta_obj)
            except Exception:
                pass
            try:
                self.ficheiro.data.reference_add(ficheiro_obj, "hasEntidades", entidade_obj)
            except Exception:
                pass

        # 5c) Ficheiro.hasEtapas: link each ficheiro to a couple of etapas from the same pasta
        etapas_by_pasta = {}
        for pasta_obj, fluxo_obj, etapa_obj, fich in etapa_objs:
            etapas_by_pasta.setdefault(pasta_obj, []).append(etapa_obj)
        for entidade_obj, pasta_obj, ficheiro_obj in all_ficheiros:
            etapas = etapas_by_pasta.get(pasta_obj, [])
            # Link up to 2 etapas per ficheiro to avoid explosion
            for etapa_obj in etapas[:2]:
                try:
                    self.ficheiro.data.reference_add(ficheiro_obj, "hasEtapas", etapa_obj)
                except Exception:
                    pass

        print("Dense connection sample data inserted and cross-linked successfully!")

        return {
            "entidades": entidade_objs,
            "pastas": [p for _, p in all_pastas],
            "ficheiros": [f for _, _, f in all_ficheiros],
            "fluxos": [f for _, _, f in fluxo_objs],
            "etapas": [e for _, _, e, _ in etapa_objs],
        }
    
    def add_fluxo(self, fluxo_name, pasta_obj=None, ficheiro_obj=None, etapa_obj=None):
        """Creates a Fluxo and links it to Pasta and Ficheiro if provided."""
        fluxo_obj = self.fluxo.data.insert({"name": fluxo_name})
        if pasta_obj:
            self.fluxo.data.reference_add(fluxo_obj, "belongsToPastas", pasta_obj)
        if ficheiro_obj:
            self.fluxo.data.reference_add(fluxo_obj, "belongsToFicheiros", ficheiro_obj)
        if etapa_obj:
            self.fluxo.data.reference_add(fluxo_obj, "hasEtapas", etapa_obj)
        return fluxo_obj

    def add_etapa(self, etapa_name, fluxo_obj=None, ficheiro_obj=None):
        """Creates an Etapa and links it to Fluxo and Ficheiro if provided."""
        etapa_obj = self.etapa.data.insert({"name": etapa_name})
        if fluxo_obj:
            self.fluxo.data.reference_add(fluxo_obj, "hasEtapas", etapa_obj)
            self.etapa.data.reference_add(etapa_obj, "belongsToFluxo", fluxo_obj)
        if ficheiro_obj:
            self.etapa.data.reference_add(etapa_obj, "hasFicheiros", ficheiro_obj)
        return etapa_obj

    def add_entidade(self, entidade_name, ficheiro_obj=None, pasta_obj=None):
        """Creates an Entidade and links it to Ficheiro and Pasta if provided."""
        entidade_obj = self.entidade.data.insert({"name": entidade_name})
        if ficheiro_obj:
            self.entidade.data.reference_add(entidade_obj, "hasFicheiros", ficheiro_obj)
        if pasta_obj:
            self.entidade.data.reference_add(entidade_obj, "hasPastas", pasta_obj)
        return entidade_obj

    def add_pasta(self, pasta_name, entidade_obj=None, fluxo_obj=None, ficheiro_obj=None):
        """Creates a Pasta and links it to Entidade, Fluxo and Ficheiro if provided."""
        pasta_obj = self.pasta.data.insert({"name": pasta_name})
        if entidade_obj:
            self.entidade.data.reference_add(entidade_obj, "hasPastas", pasta_obj)
            self.pasta.data.reference_add(pasta_obj, "hasEntidades", entidade_obj)
        if fluxo_obj:
            self.pasta.data.reference_add(pasta_obj, "hasFluxos", fluxo_obj)
        if ficheiro_obj:
            self.pasta.data.reference_add(pasta_obj, "hasFicheiros", ficheiro_obj)
        return pasta_obj

    def add_ficheiro(self, ficheiro_name, metadados_name, pasta_obj=None, entidade_obj=None, etapa_obj=None, metadados_obj=None):
        """Creates a Ficheiro and links it to Pasta, Entidade, Etapa and Metadados if provided."""
        ficheiro_obj = self.ficheiro.data.insert({"name": ficheiro_name, "metadados": metadados_name})
        if pasta_obj:
            self.pasta.data.reference_add(pasta_obj, "hasFicheiros", ficheiro_obj)
        if entidade_obj:
            self.entidade.data.reference_add(entidade_obj, "hasFicheiros", ficheiro_obj)
        if etapa_obj:
            self.etapa.data.reference_add(etapa_obj, "hasFicheiros", ficheiro_obj)
        if metadados_obj:
            self.ficheiro.data.reference_add(ficheiro_obj, "belongsToMetadados", metadados_obj)
        return ficheiro_obj
    
    def query_fluxo_etapas(self, limit=10):
        """Query Fluxos and their associated Etapas."""
        query_result = self.fluxo.query.fetch_objects(
            return_properties=["name"],
            return_references=QueryReference(
                link_on="hasEtapas",
                return_properties=["name"]
            ),
            limit=limit
        )
        
        results = []
        for obj in query_result.objects:
            fluxo_data = {
                "fluxo_name": obj.properties['name'],
                "etapas": []
            }
            
            if "hasEtapas" in obj.references:
                for etapa in obj.references["hasEtapas"].objects:
                    fluxo_data["etapas"].append(etapa.properties['name'])
            
            results.append(fluxo_data)
            
            # Print results for debugging
            print(f"Fluxo: {obj.properties['name']}")
            if "hasEtapas" in obj.references:
                for etapa in obj.references["hasEtapas"].objects:
                    print(f"  - Etapa: {etapa.properties['name']}")
            else:
                print("No 'hasEtapas' reference found for this Fluxo.")
                
        return results
    
    def query_entidade_hierarchy(self, limit=10):
        """Query Entidades and their nested hierarchy of Pastas, Ficheiros, and Metadados."""
        query_result = self.entidade.query.fetch_objects(
            return_properties=["name"],
            return_references=QueryReference(
                link_on="hasPastas",
                return_properties=["name"],
                return_references=QueryReference(
                    link_on="hasFicheiros",
                    return_properties=["name", "metadados"],
                )
            ),
            limit=limit
        )
        
        results = []
        for obj in query_result.objects:
            entidade_data = {
                "entidade_name": obj.properties['name'],
                "pastas": []
            }
            
            print(f"Entidade: {obj.properties['name']}")
            
            if "hasPastas" in obj.references:
                for pasta in obj.references["hasPastas"].objects:
                    pasta_data = {
                        "pasta_name": pasta.properties['name'],
                        "ficheiros": []
                    }
                    
                    count_ficheiros = 0
                    if "hasFicheiros" in pasta.references:
                        for ficheiro in pasta.references["hasFicheiros"].objects:
                            count_ficheiros += 1
                            ficheiro_data = {
                                "ficheiro_name": ficheiro.properties['name'],
                                "metadados": []
                            }
                            if "metadados" in ficheiro.properties:
                                ficheiro_data["metadados"].append(ficheiro.properties["metadados"])
                            
                            pasta_data["ficheiros"].append(ficheiro_data)
                    
                    print(f"  - Pasta: {pasta.properties['name'][0:7]}", end="|")
                    print(f"- Ficheiros: {count_ficheiros}")
                    
                    entidade_data["pastas"].append(pasta_data)
            
            results.append(entidade_data)
                
        return results

    def query_entidade_deep_hierarchy(self, limit=10):
        """
        Query Entidades and traverse a deep hierarchy using most available references.

        Traversal overview:
        - Entidade
          - hasPastas (Pasta)
            - hasFicheiros (Ficheiro)
              - hasEtapas (Etapa)
                - belongsToFluxo (Fluxo)
        """

        # Attempt to leverage multiple nested references at various levels
        query_result = self.entidade.query.fetch_objects(
            return_properties=["name"],
            return_references=[
                QueryReference(
                    link_on="hasPastas",
                    return_properties=["name"],
                    return_references=[
                        # Pasta -> Ficheiros (with deep refs)
                        QueryReference(
                            link_on="hasFicheiros",
                            return_properties=["name", "metadados"],
                            return_references=[
                                QueryReference(
                                    link_on="hasEtapas",
                                    return_properties=["name"],
                                    return_references=QueryReference(
                                        link_on="belongsToFluxo",
                                        return_properties=["name"],
                                    ),
                                ),
                            ],
                        ),
                        
                    ],
                )
            ],
            limit=limit,
        )

        # Structure and print results
        aggregated = []
        for ent in query_result.objects:
            ent_data = {"entidade": ent.properties.get("name"), "pastas": []}
            print(f"Entidade: {ent_data['entidade']}")

            pastas_ref = ent.references.get("hasPastas") if hasattr(ent, "references") else None
            if pastas_ref:
                for pasta in pastas_ref.objects:
                    pasta_name = pasta.properties.get("name")
                    pasta_data = {"pasta": pasta_name, "ficheiros": [], "fluxos": []}
                    print(f"  Pasta: {pasta_name}")

                    # Pasta -> Ficheiros branch
                    ficheiros_ref = pasta.references.get("hasFicheiros") if hasattr(pasta, "references") else None
                    if ficheiros_ref:
                        for fich in ficheiros_ref.objects:
                            fich_name = fich.properties.get("name")
                            fich_meta = fich.properties.get("metadados")
                            fich_data = {"ficheiro": fich_name, "metadados": fich_meta, "etapas": [], "pastas": [], "entidades": []}
                            print(f"    Ficheiro: {fich_name} | metadados={fich_meta}")

                            # Ficheiro -> hasEtapas -> belongsToFluxo
                            etapas_ref = fich.references.get("hasEtapas") if hasattr(fich, "references") else None
                            if etapas_ref:
                                for et in etapas_ref.objects:
                                    et_name = et.properties.get("name")
                                    fluxo_name = None
                                    fluxo_ref = et.references.get("belongsToFluxo") if hasattr(et, "references") else None
                                    if fluxo_ref and fluxo_ref.objects:
                                        fluxo_name = fluxo_ref.objects[0].properties.get("name")
                                    fich_data["etapas"].append({"etapa": et_name, "fluxo": fluxo_name})
                                    print(f"      Etapa: {et_name} -> Fluxo: {fluxo_name}")

                            # Ficheiro -> hasPastas, hasEntidades
                            fp_ref = fich.references.get("hasPastas") if hasattr(fich, "references") else None
                            if fp_ref:
                                for p in fp_ref.objects:
                                    fich_data["pastas"].append(p.properties.get("name"))
                            fe_ref = fich.references.get("hasEntidades") if hasattr(fich, "references") else None
                            if fe_ref:
                                for e in fe_ref.objects:
                                    fich_data["entidades"].append(e.properties.get("name"))

                            pasta_data["ficheiros"].append(fich_data)

                    # Pasta -> Fluxos branch
                    fluxos_ref = pasta.references.get("hasFluxos") if hasattr(pasta, "references") else None
                    if fluxos_ref:
                        for fl in fluxos_ref.objects:
                            fl_name = fl.properties.get("name")
                            fl_data = {"fluxo": fl_name, "etapas": [], "ficheiros": [], "pastas": []}
                            print(f"    Fluxo: {fl_name}")

                            # Fluxo -> hasEtapas
                            fl_etapas_ref = fl.references.get("hasEtapas") if hasattr(fl, "references") else None
                            if fl_etapas_ref:
                                for et in fl_etapas_ref.objects:
                                    fl_data["etapas"].append(et.properties.get("name"))

                            # Fluxo -> belongsToFicheiros
                            fl_fich_ref = fl.references.get("belongsToFicheiros") if hasattr(fl, "references") else None
                            if fl_fich_ref:
                                for ff in fl_fich_ref.objects:
                                    fl_data["ficheiros"].append(ff.properties.get("name"))

                            # Fluxo -> belongsToPastas
                            fl_pasta_ref = fl.references.get("belongsToPastas") if hasattr(fl, "references") else None
                            if fl_pasta_ref:
                                for fp in fl_pasta_ref.objects:
                                    fl_data["pastas"].append(fp.properties.get("name"))

                            pasta_data["fluxos"].append(fl_data)

                    ent_data["pastas"].append(pasta_data)

            aggregated.append(ent_data)

        return aggregated
    
    def global_semantic_search(self, query_text, alpha=0.2, limit_per_collection=5):
        """Perform a global semantic search across all collections."""
        results = []
        
        for collection_name in ["Pasta", "Fluxo", "Etapa", "Ficheiro", "Entidade"]:
            collection = self.client.collections.get(collection_name)

            search_results = collection.query.hybrid(
                query=query_text,  
                alpha=alpha,  # Set alpha for balanced results
                return_properties=["name"],  # Return these properties
                return_metadata=MetadataQuery(score=True),  # Return relevance score
                limit=limit_per_collection,  # Get top results per collection
            )

            # Store results with class name
            for obj in search_results.objects:
                results.append({
                    "class": collection_name,
                    "name": obj.properties["name"],
                    "score": obj.metadata.score,
                    "metadados": obj.properties.get("metadados", {})
                })

        # Sort results by relevance score (higher is better)
        results.sort(key=lambda x: x["score"], reverse=True)

        # Display results
        for result in results:
            print(f"[{result['class']}] {result['name']} (Score: {result['score']})")
            print(f"  Metadata: {result['metadados']}")
            
        return results
    
    def get_class_references(self, class_name: str) -> List[Dict[str, str]]:
        """Return the list of reference properties for a class from the schema.

        Output format: [{"name": <ref_property_name>, "target_collection": <target_class>}, ...]

        This inspects the collection config to extract references in a version-tolerant way
        (works if config is a dict or an object with a .references list of dataclasses).
        """
        try:
            coll = self.client.collections.get(class_name)
        except Exception:
            return []

        try:
            cfg = coll.config.get()  # v4 style returns config object/dict
        except Exception:
            # Some clients may not support .get(); try using the attribute directly
            cfg = getattr(coll, "config", None)

        refs: List[Dict[str, str]] = []
        if cfg is None:
            return refs

        # Dataclass/object-style
        obj_refs = getattr(cfg, "references", None)
        if obj_refs is not None:
            try:
                for r in obj_refs:
                    name = getattr(r, "name", None)
                    tgt = getattr(r, "target_collection", None) or getattr(r, "targetCollection", None)
                    if name and tgt:
                        refs.append({"name": name, "target_collection": tgt})
                return refs
            except Exception:
                pass

        # Dict-style
        try:
            dict_refs = cfg.get("references", []) if isinstance(cfg, dict) else []
            for r in dict_refs:
                name = r.get("name")
                tgt = r.get("target_collection") or r.get("targetCollection")
                if name and tgt:
                    refs.append({"name": name, "target_collection": tgt})
        except Exception:
            pass

        return refs

    def build_reference_plan(self, classes: List[str] | None = None, depth: int = 1) -> Dict[str, List[QueryReference]]:
        """Dynamically build a nested QueryReference plan from the schema.

        Parameters:
        - classes: list of class names to include; defaults to the 5 main classes in this project
        - depth: how deep to nest references (1 = only direct references; 2 = one hop nested, etc.)

        Returns:
        - Dict mapping class name -> List[QueryReference] suitable for return_references.
        """
        if classes is None:
            classes = ["Pasta", "Fluxo", "Etapa", "Ficheiro", "Entidade"]

        def build_for_class(cls: str, d: int) -> List[QueryReference]:
            out: List[QueryReference] = []
            for r in self.get_class_references(cls):
                nested: List[QueryReference] = build_for_class(r["target_collection"], d - 1) if d > 1 else []
                if nested:
                    out.append(
                        QueryReference(
                            link_on=r["name"],
                            return_properties=["name"],
                            return_references=nested,
                        )
                    )
                else:
                    out.append(
                        QueryReference(
                            link_on=r["name"],
                            return_properties=["name"],
                        )
                    )
            return out

        plan: Dict[str, List[QueryReference]] = {}
        for cls in classes:
            plan[cls] = build_for_class(cls, depth)
        return plan

    def _collect_fluxo_entidade_from_refs(self, obj, class_name: str, depth: int = 2):
        """Traverse returned references (guided by schema) to collect Fluxo and Entidade names.

        This uses the live schema to know, for each link_on on a class, what the target class is.
        It only traverses what's present in obj.references (so it effectively respects the plan
        used in the query), and stops at the specified depth.
        """
        collected_fluxos, collected_entidades = set(), set()

        if depth <= 0 or not hasattr(obj, "references") or obj.references is None:
            return collected_fluxos, collected_entidades

        # Map link_on -> target class for this class
        schema_refs = {r["name"]: r["target_collection"] for r in self.get_class_references(class_name)}

        for link_on, ref_obj in (obj.references or {}).items():
            target_class = schema_refs.get(link_on)
            if not target_class or not hasattr(ref_obj, "objects") or not ref_obj.objects:
                continue

            for child in ref_obj.objects:
                # Collect if child type is Fluxo or Entidade
                if target_class == "Fluxo":
                    nm = child.properties.get("name") if hasattr(child, "properties") else None
                    if nm:
                        collected_fluxos.add(nm)
                elif target_class == "Entidade":
                    nm = child.properties.get("name") if hasattr(child, "properties") else None
                    if nm:
                        collected_entidades.add(nm)

                # Recurse into child to find deeper Fluxo/Entidade
                f2, e2 = self._collect_fluxo_entidade_from_refs(child, target_class, depth - 1)
                collected_fluxos.update(f2)
                collected_entidades.update(e2)

        return collected_fluxos, collected_entidades
    
    def entity_flux_semantic_search(self, entidade_name, fluxo_name, query_text, alpha=0.2, limit=5):
        """Semantic search across collections and return Fluxo and Entidade references for explainability.

        Notes:
        - Uses correct link_on-based QueryReference definitions instead of class_name.
        - Collects both Fluxo and Entidade names via direct and nested references.
        - If entidade_name or fluxo_name are provided (non-empty), filters to results that include them.
        """
        aggregated = []
        # ["Pasta", "Fluxo", "Etapa", "Ficheiro", "Entidade"]
        for collection_name in ["Entidade"]:
            collection = self.client.collections.get(collection_name)
            plan = self.build_reference_plan(classes=[collection_name], depth=2)
            search_results = collection.query.hybrid(
                query=query_text,
                alpha=alpha,
                return_properties=["name"],
                return_metadata=MetadataQuery(score=True),
                return_references=plan.get(collection_name, []),
                limit=limit,
            )

            # Collate explainability using schema-guided traversal (respects plan depth)
            for obj in search_results.objects:
                fluxos, entidades = self._collect_fluxo_entidade_from_refs(obj, collection_name, depth=2)

                item = {
                    "class": collection_name,
                    "name": obj.properties.get("name"),
                    "score": getattr(getattr(obj, "metadata", None), "score", None),
                    "fluxos": sorted(fluxos),
                    "entidades": sorted(entidades),
                }

                # Optional filtering by provided names
                if entidade_name:
                    if entidade_name not in item["entidades"] and item["class"] != "Entidade":
                        continue
                if fluxo_name:
                    if fluxo_name not in item["fluxos"] and item["class"] != "Fluxo":
                        continue

                aggregated.append(item)

        # Sort by score descending where available
        aggregated.sort(key=lambda x: (x["score"] is not None, x["score"]), reverse=True)

        # Print compact explainability output
        for r in aggregated:
            print(f"[{r['class']}] {r['name']} (score={r['score']})")
            if r["fluxos"]:
                print(f"  Fluxos: {', '.join(r['fluxos'])}")
            if r["entidades"]:
                print(f"  Entidades: {', '.join(r['entidades'])}")

        return aggregated
    
    def close(self):
        """Close the Weaviate client connection."""
        self.client.close()


def main():
    """Main function to demonstrate the workflow."""
    # Initialize the manager
    manager = WeaviateDataManager()
    timings = {}
    
    try:
        # Setup the collections
        print("Setting up collections...")
        _t0 = time.perf_counter()
        manager.setup_collections(clean_start=True)
        timings["setup_collections"] = time.perf_counter() - _t0
        print(f"Setup collections completed in {timings['setup_collections']:.3f}s")
        
        # Insert sample data
        print("Inserting sample data...")
        _t0 = time.perf_counter()
        data_objects = manager.insert_conection_data(
            num_entidades=2,
            num_pastas_per_entidade=5,
            num_ficheiros_per_pasta=8,
            num_metadados_per_ficheiro=1,
            num_fluxos=5,
            num_etapas_per_fluxo=10
        )
        timings["insert_sample_data"] = time.perf_counter() - _t0
        print(f"Insert sample data completed in {timings['insert_sample_data']:.3f}s")
        
        # Run test queries
        print("\nRunning test queries...")
        print("\n1. Querying Fluxos and their Etapas:")
        _t0 = time.perf_counter()
        manager.query_fluxo_etapas(limit=3)
        timings["query_fluxo_etapas"] = time.perf_counter() - _t0
        print(f"Query 'fluxo_etapas' completed in {timings['query_fluxo_etapas']:.3f}s")
        
        print("\n2. Querying Entidade hierarchy:")
        _t0 = time.perf_counter()
        # manager.query_entidade_hierarchy(limit=2)
        manager.query_entidade_deep_hierarchy(limit=10)
        timings["query_entidade_hierarchy"] = time.perf_counter() - _t0
        print(f"Query 'entidade_hierarchy' completed in {timings['query_entidade_hierarchy']:.3f}s")
        
        print("\n3. Running global semantic search:")
        _t0 = time.perf_counter()
        manager.global_semantic_search("Document 1", limit_per_collection=3)
        timings["global_semantic_search"] = time.perf_counter() - _t0
        print(f"Query 'global_semantic_search' completed in {timings['global_semantic_search']:.3f}s")

        # Print timings summary
        print("\nTimings summary (seconds):")
        for k, v in timings.items():
            print(f"  - {k}: {v:.3f}s")
        
    finally:
        # Close the connection
        manager.close()
        print("\nConnection closed.")


if __name__ == "__main__":
    # main()
    manager = WeaviateDataManager(connect_to_local=True)
    try:
        manager.entity_flux_semantic_search(None, None, "contract approval process", alpha=0.3, limit=5)
    finally:
        manager.close()

def benchmark_sample_configs(
    configs: List[Dict[str, int]],
    *,
    limit_fluxo_etapas: int = 3,
    limit_entidade_hierarchy: int = 2,
    global_query: str = "Find documents about contract approvals",
    limit_per_collection: int = 3,
    alpha: float = 0.2,
    clean_start_each: bool = True,
    connect_to_local: bool = True,
) -> List[Dict[str, Any]]:
    """
    Run timed setup, data insertion, and queries for multiple sample data configurations.

    Parameters:
    - configs: List of dicts with keys matching insert_sample_data parameters:
        {"num_entidades", "num_pastas_per_entidade", "num_ficheiros_per_pasta",
         "num_metadados_per_ficheiro", "num_fluxos", "num_etapas_per_fluxo"}
    - limit_fluxo_etapas: Limit for query_fluxo_etapas
    - limit_entidade_hierarchy: Limit for query_entidade_hierarchy
    - global_query: Query text for global_semantic_search
    - limit_per_collection: Limit per collection for global_semantic_search
    - clean_start_each: If True, recreate collections for each config (timed). If False, reuse schema.
    - connect_to_local: Whether to connect to local Weaviate

    Returns:
    - List of dicts: [{"config": <input_config>, "timings": {step: seconds, ...}}]
    """
    results: List[Dict[str, Any]] = []
    manager = WeaviateDataManager(connect_to_local=connect_to_local)
    try:
        for idx, cfg in enumerate(configs, start=1):
            print(f"\n=== Benchmark {idx}/{len(configs)} ===")
            print(
                "Config: "
                f"entidades={cfg.get('num_entidades')}, "
                f"pastas/entidade={cfg.get('num_pastas_per_entidade')}, "
                f"ficheiros/pasta={cfg.get('num_ficheiros_per_pasta')}, "
                f"metadados/ficheiro={cfg.get('num_metadados_per_ficheiro')}, "
                f"fluxos={cfg.get('num_fluxos')}, "
                f"etapas/fluxo={cfg.get('num_etapas_per_fluxo')}"
            )

            timings: Dict[str, float] = {}

            # Setup schema (optionally clean each iteration)
            if clean_start_each or idx == 1:
                print("Setting up collections...")
                _t0 = time.perf_counter()
                manager.setup_collections(clean_start=True)
                timings["setup_collections"] = time.perf_counter() - _t0
                print(f"Setup collections completed in {timings['setup_collections']:.3f}s")

            # Insert sample data with provided configuration
            print("Inserting sample data...")
            _t0 = time.perf_counter()
            manager.insert_conection_data(
                num_entidades=cfg.get("num_entidades", 2),
                num_pastas_per_entidade=cfg.get("num_pastas_per_entidade", 5),
                num_ficheiros_per_pasta=cfg.get("num_ficheiros_per_pasta", 8),
                num_metadados_per_ficheiro=cfg.get("num_metadados_per_ficheiro", 1),
                num_fluxos=cfg.get("num_fluxos", 5),
                num_etapas_per_fluxo=cfg.get("num_etapas_per_fluxo", 10),
            )
            timings["insert_sample_data"] = time.perf_counter() - _t0
            print(f"Insert sample data completed in {timings['insert_sample_data']:.3f}s")

            # Queries
            print("\nRunning test queries...")
            print("1. Querying Fluxos and their Etapas:")
            _t0 = time.perf_counter()
            manager.query_fluxo_etapas(limit=limit_fluxo_etapas)
            timings["query_fluxo_etapas"] = time.perf_counter() - _t0
            print(f"Query 'fluxo_etapas' completed in {timings['query_fluxo_etapas']:.3f}s")

            print("\n2. Querying Entidade hierarchy:")
            _t0 = time.perf_counter()
            manager.query_entidade_deep_hierarchy(limit=limit_entidade_hierarchy)
            timings["query_entidade_deep_hierarchy"] = time.perf_counter() - _t0
            print(f"Query 'entidade_hierarchy' completed in {timings['query_entidade_deep_hierarchy']:.3f}s")

            print("\n3. Running global semantic (hybrid) search:")
            _t0 = time.perf_counter()
            manager.global_semantic_search(global_query, alpha=alpha, limit_per_collection=limit_per_collection)
            timings["global_semantic_search"] = time.perf_counter() - _t0
            print(f"Query 'global_semantic_search' completed in {timings['global_semantic_search']:.3f}s")

            # Summary for this config
            print("\nTimings summary (seconds):")
            for k, v in timings.items():
                print(f"  - {k}: {v:.3f}s")

            results.append({"config": dict(cfg), "timings": timings})
    finally:
        manager.close()
        print("\nConnection closed.")

    # Print final aggregated summary table
    print("\n=== Aggregated results ===")
    for i, item in enumerate(results, start=1):
        cfg = item["config"]
        t = item["timings"]
        print(
            f"[{i}] ent={cfg.get('num_entidades')}, "
            f"pastas/ent={cfg.get('num_pastas_per_entidade')}, "
            f"ficheiros/pasta={cfg.get('num_ficheiros_per_pasta')}, "
            f"fluxos={cfg.get('num_fluxos')}, etapas/fluxo={cfg.get('num_etapas_per_fluxo')} | "
            f"setup={t.get('setup_collections', 0):.3f}s, "
            f"insert={t.get('insert_sample_data', 0):.3f}s, "
            f"q1={t.get('query_fluxo_etapas', 0):.3f}s, "
            f"q2={t.get('query_entidade_hierarchy', 0):.3f}s, "
            f"q3={t.get('global_semantic_search', 0):.3f}s"
        )

    return results