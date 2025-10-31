import weaviate
import os
import time
import csv
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from weaviate.classes.config import Property, DataType, ReferenceProperty, Configure
from weaviate.classes.query import QueryReference, MetadataQuery
from typing import List, Dict, Any, Optional


class WeaviateDataManager:
    # Define Ollama API details
    OLLAMA_API_ENDPOINT = "http://host.docker.internal:11434"
    VECTOR_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    GENERATION_MODEL = "qwen2.5:latest"

    def __init__(self, connect_to_local: bool = True, max_workers: Optional[int] = None):
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

        # Concurrency controls (default sensible for I/O-bound work). Cap to avoid overload.
        default_workers = max(4, (os.cpu_count() or 4))
        self.max_workers = max_workers or min(default_workers, 16)

    # --- small retry helper for transient connection issues ---
    def _with_retries(self, fn, *args, retries: int = 3, base_delay: float = 0.2, **kwargs):
        last_err: Exception = Exception("Retry attempts exhausted")
        for attempt in range(retries):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                last_err = e
                # exponential backoff
                time.sleep(base_delay * (2 ** attempt))
        # re-raise last error if all retries failed
        raise last_err

    def setup_collections(self, clean_start: bool = True):
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
            except Exception:
                pass

    def _create_collections(self):
        """Create all required collections."""
        # 1. Create "Fluxo" Collection
        self._create_collection(
            "Fluxo",
            "A workflow that contains multiple stages (etapas).",
            [Property(name="name", data_type=DataType.TEXT)],
        )

        # 2. Create "Etapa" Collection
        self._create_collection(
            "Etapa",
            "A stage within a workflow.",
            [Property(name="name", data_type=DataType.TEXT)],
        )

        # 3. Create "Entidade" Collection
        self._create_collection(
            "Entidade",
            "An entity that owns folders (pastas).",
            [Property(name="name", data_type=DataType.TEXT)],
        )

        # 4. Create "Pasta" Collection
        self._create_collection(
            "Pasta",
            "A folder belonging to an entity and containing documents (ficheiros).",
            [Property(name="name", data_type=DataType.TEXT)],
        )

        # 5. Create "Ficheiro" Collection
        self._create_collection(
            "Ficheiro",
            "A document that contains metadata.",
            [Property(name="name", data_type=DataType.TEXT), Property(name="metadados", data_type=DataType.TEXT)],
        )

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
                model=self.GENERATION_MODEL,
            ),
        )

    def _get_collection_objects(self):
        """Get references to all collection objects."""
        self.fluxo = self.client.collections.get("Fluxo")
        self.etapa = self.client.collections.get("Etapa")
        self.entidade = self.client.collections.get("Entidade")
        self.pasta = self.client.collections.get("Pasta")
        self.ficheiro = self.client.collections.get("Ficheiro")

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

        self.ficheiro.config.add_reference(ReferenceProperty(name="hasEtapas", target_collection="Etapa"))
        self.ficheiro.config.add_reference(ReferenceProperty(name="hasPastas", target_collection="Pasta"))
        self.ficheiro.config.add_reference(ReferenceProperty(name="hasEntidades", target_collection="Entidade"))

    def insert_sample_data(
        self,
        num_entidades=2,
        num_pastas_per_entidade=5,
        num_ficheiros_per_pasta=8,
        num_metadados_per_ficheiro=1,
        num_fluxos=5,
        num_etapas_per_fluxo=10,
    ):
        """Insert sample data into all collections with proper references."""
        entidade_objs = [self.add_entidade(f"Empresa {i+1}") for i in range(num_entidades)]

        pasta_objs = []
        for i, entidade_obj in enumerate(entidade_objs):
            for j in range(num_pastas_per_entidade):
                pasta_obj = self.add_pasta(f"Pasta {j+1} of Empresa {i+1}", entidade_obj=entidade_obj)
                pasta_objs.append(pasta_obj)

        ficheiro_objs = []
        for i, pasta_obj in enumerate(pasta_objs):
            entidade_obj = entidade_objs[i % len(entidade_objs)]
            for j in range(num_ficheiros_per_pasta):
                ficheiro_obj = self.add_ficheiro(
                    f"Documento {j+1} in {pasta_obj}", f"Metadata {j+1}", pasta_obj=pasta_obj, entidade_obj=entidade_obj
                )
                ficheiro_objs.append(ficheiro_obj)

        fluxo_objs = []
        for i in range(num_fluxos):
            pasta_obj = pasta_objs[i % len(pasta_objs)]
            ficheiro_obj = ficheiro_objs[i % len(ficheiro_objs)]
            fluxo_obj = self.add_fluxo(f"Fluxo {i+1}", pasta_obj=pasta_obj, ficheiro_obj=ficheiro_obj)
            fluxo_objs.append(fluxo_obj)

        etapa_objs = []
        for i, fluxo_obj in enumerate(fluxo_objs):
            ficheiro_obj = ficheiro_objs[i % len(ficheiro_objs)]
            for j in range(num_etapas_per_fluxo):
                etapa_obj = self.add_etapa(
                    f"Etapa {j+1} of Fluxo {i+1}", fluxo_obj=fluxo_obj, ficheiro_obj=ficheiro_obj
                )
                etapa_objs.append(etapa_obj)

        print("Bulk sample data inserted and linked successfully!")

        return {
            "entidades": entidade_objs,
            "pastas": pasta_objs,
            "ficheiros": ficheiro_objs,
            "fluxos": fluxo_objs,
            "etapas": etapa_objs,
        }

    def insert_conection_data(
        self,
        num_entidades: int = 2,
        num_pastas_per_entidade: int = 5,
        num_ficheiros_per_pasta: int = 8,
        num_metadados_per_ficheiro: int = 1,
        num_fluxos: int = 5,
        num_etapas_per_fluxo: int = 10,
    ):
        """
        Parallelized: Insert data with a denser web of connections, maximizing references while using a thread pool.
        """
        # 1) Create Entidades and Pastas
        entidade_objs = [self.add_entidade(f"Empresa {i+1}") for i in range(num_entidades)]

        pasta_objs_by_entidade = []  # List[List[pasta]] aligned with entidade index
        all_pastas = []
        for i, entidade_obj in enumerate(entidade_objs):
            pastas_for_entidade = [
                self.add_pasta(f"Pasta {j+1} of Empresa {i+1}", entidade_obj=entidade_obj)
                for j in range(num_pastas_per_entidade)
            ]
            for p in pastas_for_entidade:
                all_pastas.append((entidade_obj, p))
            pasta_objs_by_entidade.append(pastas_for_entidade)

        # 2) Create Ficheiros for each Pasta (link to Pasta + Entidade) in parallel
        ficheiros_by_pasta: Dict[Any, List[Any]] = {}
        all_ficheiros: List[Any] = []

        def _create_ficheiro(entidade_obj, pasta_obj, k):
            return self._with_retries(
                self.add_ficheiro,
                ficheiro_name=f"Documento {k+1} in {pasta_obj}",
                metadados_name=f"Metadata {k+1}",
                pasta_obj=pasta_obj,
                entidade_obj=entidade_obj,
            )

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = {}
            for entidade_obj, pasta_obj in all_pastas:
                for k in range(num_ficheiros_per_pasta):
                    fut = ex.submit(_create_ficheiro, entidade_obj, pasta_obj, k)
                    futures[fut] = (entidade_obj, pasta_obj)

            for fut in as_completed(futures):
                entidade_obj, pasta_obj = futures[fut]
                ficheiro_obj = fut.result()
                ficheiros_by_pasta.setdefault(pasta_obj, []).append(ficheiro_obj)
                all_ficheiros.append((entidade_obj, pasta_obj, ficheiro_obj))

        # 3) Create Fluxos in parallel
        fluxo_objs: List[Any] = []
        total_pastas = len(all_pastas)

        def _create_fluxo(i: int):
            entidade_obj, pasta_obj = all_pastas[i % total_pastas]
            pasta_ficheiros = ficheiros_by_pasta[pasta_obj]
            primary_fich = pasta_ficheiros[i % len(pasta_ficheiros)]
            fluxo_obj = self._with_retries(
                self.add_fluxo,
                fluxo_name=f"Fluxo {i+1}", pasta_obj=pasta_obj, ficheiro_obj=primary_fich
            )
            return (entidade_obj, pasta_obj, fluxo_obj)

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            fluxo_objs.extend(list(ex.map(_create_fluxo, range(num_fluxos))))

        # 4) Create Etapas for each Fluxo in parallel (per fluxo)
        etapa_objs: List[Any] = []

        def _create_etapas_for_fluxo(i: int, entidade_obj, pasta_obj, fluxo_obj):
            local = []
            pasta_ficheiros = ficheiros_by_pasta[pasta_obj]
            for j in range(num_etapas_per_fluxo):
                fich_for_etapa = pasta_ficheiros[(i + j) % len(pasta_ficheiros)]
                etapa_obj = self._with_retries(
                    self.add_etapa,
                    etapa_name=f"Etapa {j+1} of Fluxo {i+1}",
                    fluxo_obj=fluxo_obj,
                    ficheiro_obj=fich_for_etapa,
                )
                local.append((pasta_obj, fluxo_obj, etapa_obj, fich_for_etapa))
            return local

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = [ex.submit(_create_etapas_for_fluxo, i, ent, p, fl) for i, (ent, p, fl) in enumerate(fluxo_objs)]
            for fut in as_completed(futures):
                etapa_objs.extend(fut.result())

        # 5) Add reverse/cross links in parallel
        def _link_pasta_fluxo(pasta_obj, fluxo_obj):
            try:
                self._with_retries(self.pasta.data.reference_add, pasta_obj, "hasFluxos", fluxo_obj)
            except Exception:
                pass

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            list(ex.map(lambda t: _link_pasta_fluxo(t[1], t[2]), fluxo_objs))

        def _link_ficheiro_pasta_entidade(entidade_obj, pasta_obj, ficheiro_obj):
            try:
                self._with_retries(self.ficheiro.data.reference_add, ficheiro_obj, "hasPastas", pasta_obj)
            except Exception:
                pass
            try:
                self._with_retries(self.ficheiro.data.reference_add, ficheiro_obj, "hasEntidades", entidade_obj)
            except Exception:
                pass

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            list(ex.map(lambda t: _link_ficheiro_pasta_entidade(*t), all_ficheiros))

        etapas_by_pasta: Dict[Any, List[Any]] = {}
        for pasta_obj, fluxo_obj, etapa_obj, fich in etapa_objs:
            etapas_by_pasta.setdefault(pasta_obj, []).append(etapa_obj)

        def _link_ficheiro_etapas(entidade_obj, pasta_obj, ficheiro_obj):
            etapas = etapas_by_pasta.get(pasta_obj, [])
            for etapa_obj in etapas[:2]:
                try:
                    self._with_retries(self.ficheiro.data.reference_add, ficheiro_obj, "hasEtapas", etapa_obj)
                except Exception:
                    pass

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            list(ex.map(lambda t: _link_ficheiro_etapas(*t), all_ficheiros))

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

    def add_metadados(self, metadados_data, ficheiro_obj=None, etapa_obj=None, pasta_obj=None, entidade_obj=None):
        """Creates Metadados and links it to Ficheiro, Etapa, Pasta and Entidade if provided."""
        metadados_obj = self.metadados.data.insert({"name": metadados_data})
        if ficheiro_obj:
            self.metadados.data.reference_add(metadados_obj, "hasFicheiros", ficheiro_obj)
        if etapa_obj:
            self.metadados.data.reference_add(metadados_obj, "hasEtapas", etapa_obj)
        if pasta_obj:
            self.metadados.data.reference_add(metadados_obj, "hasPastas", pasta_obj)
        if entidade_obj:
            self.metadados.data.reference_add(metadados_obj, "hasEntidades", entidade_obj)
        return metadados_obj

    def query_fluxo_etapas(self, limit=10):
        """Query Fluxos and their associated Etapas."""
        query_result = self.fluxo.query.fetch_objects(
            return_properties=["name"],
            return_references=QueryReference(link_on="hasEtapas", return_properties=["name"]),
            limit=limit,
        )

        results = []
        for obj in query_result.objects:
            fluxo_data = {"fluxo_name": obj.properties["name"], "etapas": []}

            if "hasEtapas" in obj.references:
                for etapa in obj.references["hasEtapas"].objects:
                    fluxo_data["etapas"].append(etapa.properties["name"])

            results.append(fluxo_data)

            # print(f"Fluxo: {obj.properties['name']}")
            if "hasEtapas" in obj.references:
                for etapa in obj.references["hasEtapas"].objects:
                    # print(f"  - Etapa: {etapa.properties['name']}")
                    pass
            else:
                # print("No 'hasEtapas' reference found for this Fluxo.")
                pass

        return results

    def query_entidade_deep_hierarchy(self, limit=10):
        """Deep hierarchy query (same as baseline)."""
        query_result = self.entidade.query.fetch_objects(
            return_properties=["name"],
            return_references=[
                QueryReference(
                    link_on="hasPastas",
                    return_properties=["name"],
                    return_references=[
                        QueryReference(
                            link_on="hasFicheiros",
                            return_properties=["name", "metadados"],
                            return_references=[
                                QueryReference(
                                    link_on="hasEtapas",
                                    return_properties=["name"],
                                    return_references=QueryReference(
                                        link_on="belongsToFluxo", return_properties=["name"]
                                    ),
                                )
                            ],
                        )
                    ],
                )
            ],
            limit=limit,
        )

        aggregated = []
        for ent in query_result.objects:
            ent_data = {"entidade": ent.properties.get("name"), "pastas": []}
            # print(f"Entidade: {ent_data['entidade']}")

            pastas_ref = ent.references.get("hasPastas") if hasattr(ent, "references") else None
            if pastas_ref:
                for pasta in pastas_ref.objects:
                    pasta_name = pasta.properties.get("name")
                    pasta_data = {"pasta": pasta_name, "ficheiros": [], "fluxos": []}
                    # print(f"  Pasta: {pasta_name}")

                    ficheiros_ref = pasta.references.get("hasFicheiros") if hasattr(pasta, "references") else None
                    if ficheiros_ref:
                        for fich in ficheiros_ref.objects:
                            fich_name = fich.properties.get("name")
                            fich_meta = fich.properties.get("metadados")
                            fich_data = {
                                "ficheiro": fich_name,
                                "metadados": fich_meta,
                                "etapas": [],
                                "pastas": [],
                                "entidades": [],
                            }
                            # print(f"    Ficheiro: {fich_name} | metadados={fich_meta}")

                            etapas_ref = fich.references.get("hasEtapas") if hasattr(fich, "references") else None
                            if etapas_ref:
                                for et in etapas_ref.objects:
                                    et_name = et.properties.get("name")
                                    fluxo_name = None
                                    fluxo_ref = et.references.get("belongsToFluxo") if hasattr(et, "references") else None
                                    if fluxo_ref and fluxo_ref.objects:
                                        fluxo_name = fluxo_ref.objects[0].properties.get("name")
                                    fich_data["etapas"].append({"etapa": et_name, "fluxo": fluxo_name})
                                    # print(f"      Etapa: {et_name} -> Fluxo: {fluxo_name}")

                            fp_ref = fich.references.get("hasPastas") if hasattr(fich, "references") else None
                            if fp_ref:
                                for p in fp_ref.objects:
                                    fich_data["pastas"].append(p.properties.get("name"))
                            fe_ref = fich.references.get("hasEntidades") if hasattr(fich, "references") else None
                            if fe_ref:
                                for e in fe_ref.objects:
                                    fich_data["entidades"].append(e.properties.get("name"))

                            pasta_data["ficheiros"].append(fich_data)

                    fluxos_ref = pasta.references.get("hasFluxos") if hasattr(pasta, "references") else None
                    if fluxos_ref:
                        for fl in fluxos_ref.objects:
                            fl_name = fl.properties.get("name")
                            fl_data = {"fluxo": fl_name, "etapas": [], "ficheiros": [], "pastas": []}
                            # print(f"    Fluxo: {fl_name}")

                            fl_etapas_ref = fl.references.get("hasEtapas") if hasattr(fl, "references") else None
                            if fl_etapas_ref:
                                for et in fl_etapas_ref.objects:
                                    fl_data["etapas"].append(et.properties.get("name"))

                            fl_fich_ref = fl.references.get("belongsToFicheiros") if hasattr(fl, "references") else None
                            if fl_fich_ref:
                                for ff in fl_fich_ref.objects:
                                    fl_data["ficheiros"].append(ff.properties.get("name"))

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
                alpha=alpha,
                return_properties=["name"],
                return_metadata=MetadataQuery(score=True),
                limit=limit_per_collection,
            )

            for obj in search_results.objects:
                results.append(
                    {
                        "class": collection_name,
                        "name": obj.properties["name"],
                        "score": obj.metadata.score,
                        "metadados": obj.properties.get("metadados", {}),
                    }
                )

        results.sort(key=lambda x: x["score"], reverse=True)

        for result in results:
            # print(f"[{result['class']}] {result['name']} (Score: {result['score']})")
            # print(f"  Metadata: {result['metadados']}")
            pass

        return results

    def close(self):
        """Close the Weaviate client connection."""
        self.client.close()


def benchmark_sample_configs(
    configs: List[Dict[str, int]],
    *,
    limit_fluxo_etapas: int = 3,
    limit_entidade_hierarchy: int = 2,
    global_query: str = "Find documents about contract approvals",
    limit_per_collection: int = 3,
    clean_start_each: bool = True,
    connect_to_local: bool = True,
    max_workers: Optional[int] = None,
    csv_path: Optional[os.PathLike | str] = None,
) -> List[Dict[str, Any]]:
    """
    Optimized version: Run timed setup, data insertion, and queries for multiple sample data configurations.
    """
    results: List[Dict[str, Any]] = []

    # Prepare CSV appender if requested
    csv_file: Optional[Path] = None
    fieldnames = [
        "num_entidades",
        "num_pastas_per_entidade",
        "num_ficheiros_per_pasta",
        "num_metadados_per_ficheiro",
        "num_fluxos",
        "num_etapas_per_fluxo",
        "setup_collections",
        "insert_sample_data",
        "query_fluxo_etapas",
        "query_entidade_hierarchy",
        "global_semantic_search",
    ]

    if csv_path:
        csv_file = Path(csv_path).resolve()
        csv_file.parent.mkdir(parents=True, exist_ok=True)
        # Write header once if file does not exist
        if not csv_file.exists():
            with csv_file.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
    manager = WeaviateDataManager(connect_to_local=connect_to_local, max_workers=max_workers)
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

            if clean_start_each or idx == 1:
                print("Setting up collections...")
                _t0 = time.perf_counter()
                manager.setup_collections(clean_start=True)
                timings["setup_collections"] = time.perf_counter() - _t0
                print(f"Setup collections completed in {timings['setup_collections']:.3f}s")

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

            print("\nRunning test queries...")
            print("1. Querying Fluxos and their Etapas:")
            _t0 = time.perf_counter()
            manager.query_fluxo_etapas(limit=limit_fluxo_etapas)
            timings["query_fluxo_etapas"] = time.perf_counter() - _t0
            print(f"Query 'fluxo_etapas' completed in {timings['query_fluxo_etapas']:.3f}s")

            print("\n2. Querying Entidade hierarchy:")
            _t0 = time.perf_counter()
            manager.query_entidade_deep_hierarchy(limit=limit_entidade_hierarchy)
            timings["query_entidade_hierarchy"] = time.perf_counter() - _t0
            print(f"Query 'entidade_hierarchy' completed in {timings['query_entidade_hierarchy']:.3f}s")

            print("\n3. Running global semantic search:")
            _t0 = time.perf_counter()
            manager.global_semantic_search(global_query, limit_per_collection=limit_per_collection)
            timings["global_semantic_search"] = time.perf_counter() - _t0
            print(f"Query 'global_semantic_search' completed in {timings['global_semantic_search']:.3f}s")

            print("\nTimings summary (seconds):")
            for k, v in timings.items():
                print(f"  - {k}: {v:.3f}s")

            item = {"config": dict(cfg), "timings": timings}
            results.append(item)

            # Append to CSV immediately if requested
            if csv_file is not None:
                cfg = item["config"]
                t = item["timings"]
                row = {
                    "num_entidades": cfg.get("num_entidades"),
                    "num_pastas_per_entidade": cfg.get("num_pastas_per_entidade"),
                    "num_ficheiros_per_pasta": cfg.get("num_ficheiros_per_pasta"),
                    "num_metadados_per_ficheiro": cfg.get("num_metadados_per_ficheiro"),
                    "num_fluxos": cfg.get("num_fluxos"),
                    "num_etapas_per_fluxo": cfg.get("num_etapas_per_fluxo"),
                    "setup_collections": f"{t.get('setup_collections', 0.0):.6f}",
                    "insert_sample_data": f"{t.get('insert_sample_data', 0.0):.6f}",
                    "query_fluxo_etapas": f"{t.get('query_fluxo_etapas', 0.0):.6f}",
                    "query_entidade_hierarchy": f"{t.get('query_entidade_hierarchy', 0.0):.6f}",
                    "global_semantic_search": f"{t.get('global_semantic_search', 0.0):.6f}",
                }
                with csv_file.open("a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writerow(row)
            
        print("\n=== Aggregated results (optimized) ===")
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
    finally:
        manager.close()
        print("\nConnection closed.")

    print("\n=== Aggregated results (optimized) ===")
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
