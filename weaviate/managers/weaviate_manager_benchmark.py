"""
Journalism example: structured schema, a real report, and targeted structured queries.

This file is a runnable script:
- Creates a schema for a newsroom
- Inserts a real report and links it to entities and a topic
- Answers structured questions via graph traversal, not semantic search
"""

from typing import Any, Dict, List, Optional
import weaviate
from weaviate.classes.config import DataType, Property, ReferenceProperty, Configure
from weaviate.classes.query import QueryReference, MetadataQuery


class NewsWeaviate:
    def __init__(self, *, connect_to_local: bool = True) -> None:
        # Connect to local Weaviate (adapt if needed)
        self.client = weaviate.connect_to_local() if connect_to_local else weaviate.connect_to_local()
        # Optional: configure generative provider (used by _create_collection)
        self.OLLAMA_API_ENDPOINT = "http://host.docker.internal:11434"
        self.GENERATION_MODEL = "qwen2.5:latest"

    # -------------------------- Schema --------------------------
    def reset_schema(self) -> None:
        """Drop and recreate a minimal schema tailored for a newsroom."""
        for name in ["Etapa", "Ficheiro", "Fluxo", "Entidade"]:
            try:
                self.client.collections.delete(name)
            except Exception:
                pass

        # Entidade: a named entity (country, person, org) optionally with a role
        self._create_collection(
            name="Entidade",
            description="Named entity such as a country, person, or organization.",
            properties=[
                Property(name="name", data_type=DataType.TEXT),
                Property(name="kind", data_type=DataType.TEXT),  # Country | Person | Organization
                Property(name="role", data_type=DataType.TEXT),  # e.g., Journalist
            ],
            source_properties=["name", "kind", "role"],
        )

        # Ficheiro: a news report
        self._create_collection(
            name="Ficheiro",
            description="News report (file/document) with title, body, and date.",
            properties=[
                Property(name="title", data_type=DataType.TEXT),
                Property(name="body", data_type=DataType.TEXT),
                Property(name="date", data_type=DataType.DATE),  # ISO-8601 (YYYY-MM-DD)
            ],
            source_properties=["title", "body"],
        )

        # Fluxo: a topic/storyline being followed (e.g., Russia vs Ukraine War)
        self._create_collection(
            name="Fluxo",
            description="Topic or storyline with multiple stages and reports.",
            properties=[
                Property(name="name", data_type=DataType.TEXT),
                Property(name="description", data_type=DataType.TEXT),
            ],
            source_properties=["name", "description"],
        )

        # Etapa: a stage/event in a topic, often caused or evidenced by a report
        self._create_collection(
            name="Etapa",
            description="Event/stage in a topic with date and category.",
            properties=[
                Property(name="name", data_type=DataType.TEXT),
                Property(name="date", data_type=DataType.DATE),
                Property(name="description", data_type=DataType.TEXT),
                Property(name="category", data_type=DataType.TEXT),  # e.g., war_crime, attack, diplomacy
            ],
            source_properties=["name", "description", "category"],
        )

        # Add references
        ent = self.client.collections.get("Entidade")
        ent.config.add_reference(ReferenceProperty(name="hasReports", target_collection="Ficheiro"))

        fich = self.client.collections.get("Ficheiro")
        fich.config.add_reference(ReferenceProperty(name="hasEntidades", target_collection="Entidade"))
        fich.config.add_reference(ReferenceProperty(name="partOfFlux", target_collection="Fluxo"))
        fich.config.add_reference(ReferenceProperty(name="triggersEtapas", target_collection="Etapa"))

        flx = self.client.collections.get("Fluxo")
        flx.config.add_reference(ReferenceProperty(name="hasStages", target_collection="Etapa"))
        flx.config.add_reference(ReferenceProperty(name="hasReports", target_collection="Ficheiro"))

        etp = self.client.collections.get("Etapa")
        etp.config.add_reference(ReferenceProperty(name="belongsToFlux", target_collection="Fluxo"))
        etp.config.add_reference(ReferenceProperty(name="aboutReports", target_collection="Ficheiro"))
        etp.config.add_reference(ReferenceProperty(name="aboutEntities", target_collection="Entidade"))

        print("Schema created: Entidade, Ficheiro, Fluxo, Etapa with references.")

    def _create_collection(self, name: str, description: str, properties: List[Property], *, source_properties: List[str]) -> None:
        """Create a collection with vectorization & Ollama generative config.

        Uses NamedVectors with text2vec-transformers module and masks pooling. Adjust source_properties as needed.
        """
        self.client.collections.create(
            name=name,
            description=description,
            properties=properties,
            vectorizer_config=[
                Configure.NamedVectors.text2vec_transformers(
                    name="text_vector",
                    source_properties=source_properties,
                    pooling_strategy="masked_mean",
                )
            ],
            generative_config=Configure.Generative.ollama(
                api_endpoint=self.OLLAMA_API_ENDPOINT,
                model=self.GENERATION_MODEL,
            ),
        )

    # -------------------------- Inserts --------------------------
    def create_entity(self, name: str, kind: str, role: Optional[str] = None):
        return self.client.collections.get("Entidade").data.insert({"name": name, "kind": kind, "role": role or ""})

    def create_topic(self, name: str, description: str = ""):
        return self.client.collections.get("Fluxo").data.insert({"name": name, "description": description})

    def create_report(self, title: str, body: str, date_iso: str):
        return self.client.collections.get("Ficheiro").data.insert({"title": title, "body": body, "date": date_iso})

    def create_stage(self, name: str, date_iso: str, description: str, category: str):
        return self.client.collections.get("Etapa").data.insert({
            "name": name,
            "date": date_iso,
            "description": description,
            "category": category,
        })

    # -------------------------- Linking --------------------------
    def link_report_to_entities(self, report_obj, entity_objs: List[Any]) -> None:
        ficheiro = self.client.collections.get("Ficheiro")
        entidade = self.client.collections.get("Entidade")
        for ent in entity_objs:
            try:
                ficheiro.data.reference_add(report_obj, "hasEntidades", ent)
            except Exception:
                pass
            try:
                entidade.data.reference_add(ent, "hasReports", report_obj)
            except Exception:
                pass

    def link_report_to_topic(self, report_obj, topic_obj) -> None:
        ficheiro = self.client.collections.get("Ficheiro")
        fluxo = self.client.collections.get("Fluxo")
        try:
            ficheiro.data.reference_add(report_obj, "partOfFlux", topic_obj)
        except Exception:
            pass
        try:
            fluxo.data.reference_add(topic_obj, "hasReports", report_obj)
        except Exception:
            pass

    def link_stage_to_topic_and_report_and_entities(self, stage_obj, topic_obj, report_obj, entities: List[Any]) -> None:
        etapa = self.client.collections.get("Etapa")
        fluxo = self.client.collections.get("Fluxo")
        ficheiro = self.client.collections.get("Ficheiro")
        # Stage -> Flux
        try:
            etapa.data.reference_add(stage_obj, "belongsToFlux", topic_obj)
        except Exception:
            pass
        try:
            fluxo.data.reference_add(topic_obj, "hasStages", stage_obj)
        except Exception:
            pass

        # Stage <-> Report
        try:
            etapa.data.reference_add(stage_obj, "aboutReports", report_obj)
        except Exception:
            pass
        try:
            ficheiro.data.reference_add(report_obj, "triggersEtapas", stage_obj)
        except Exception:
            pass

        # Stage -> Entities (optional for richer traversal)
        for ent in entities:
            try:
                etapa.data.reference_add(stage_obj, "aboutEntities", ent)
            except Exception:
                pass

    # -------------------------- Structured queries --------------------------
    def topic_status(self, topic_name: str) -> Dict[str, Any]:
        """Return latest stages and counts for a topic (Fluxo), using structured traversal and client-side sorting."""
        # Fetch topic with stages and reports
        res = self.client.collections.get("Fluxo").query.fetch_objects(
            return_properties=["name", "description"],
            return_references=[
                QueryReference(link_on="hasStages", return_properties=["name", "date", "category", "description"]),
                QueryReference(link_on="hasReports", return_properties=["title", "date"]),
            ],
            limit=50,
        )

        topic_obj = None
        for obj in getattr(res, "objects", []) or []:
            if obj.properties.get("name") == topic_name:
                topic_obj = obj
                break
        if not topic_obj:
            return {"topic": topic_name, "exists": False}

        # Gather stages and sort by date desc
        stages = []
        st_ref = getattr(topic_obj, "references", {}).get("hasStages") if hasattr(topic_obj, "references") else None
        if st_ref:
            for st in st_ref.objects:
                stages.append({
                    "name": st.properties.get("name"),
                    "date": st.properties.get("date"),
                    "category": st.properties.get("category"),
                    "description": st.properties.get("description"),
                })
        stages.sort(key=lambda x: x.get("date") or "", reverse=True)

        # Count reports
        report_count = 0
        rp_ref = getattr(topic_obj, "references", {}).get("hasReports") if hasattr(topic_obj, "references") else None
        if rp_ref:
            report_count = len(rp_ref.objects or [])

        latest = stages[0] if stages else None
        return {
            "topic": topic_obj.properties.get("name"),
            "description": topic_obj.properties.get("description"),
            "exists": True,
            "report_count": report_count,
            "latest_stage": latest,
            "stages": stages[:5],  # top 5 for brevity
        }

    def latest_war_crime(self, topic_name: str) -> Optional[Dict[str, Any]]:
        """Return the latest stage categorized as a war crime under the topic, if any."""
        res = self.client.collections.get("Fluxo").query.fetch_objects(
            return_properties=["name"],
            return_references=[QueryReference(link_on="hasStages", return_properties=["name", "date", "category", "description"])],
            limit=50,
        )
        topic_obj = None
        for obj in getattr(res, "objects", []) or []:
            if obj.properties.get("name") == topic_name:
                topic_obj = obj
                break
        if not topic_obj:
            return None

        stages = []
        st_ref = getattr(topic_obj, "references", {}).get("hasStages") if hasattr(topic_obj, "references") else None
        if st_ref:
            for st in st_ref.objects:
                if (st.properties.get("category") or "").lower() == "war_crime":
                    stages.append({
                        "name": st.properties.get("name"),
                        "date": st.properties.get("date"),
                        "category": st.properties.get("category"),
                        "description": st.properties.get("description"),
                    })
        stages.sort(key=lambda x: x.get("date") or "", reverse=True)
        return stages[0] if stages else None

    def reporters_for_latest_stage(self, topic_name: str) -> List[str]:
        """Find the journalist(s) who reported the report linked to the latest stage in the topic."""
        # Get latest stage for the topic
        res = self.client.collections.get("Fluxo").query.fetch_objects(
            return_properties=["name"],
            return_references=[QueryReference(link_on="hasStages", return_properties=["name", "date"],)],
            limit=50,
        )
        topic_obj = None
        for obj in getattr(res, "objects", []) or []:
            if obj.properties.get("name") == topic_name:
                topic_obj = obj
                break
        if not topic_obj:
            return []

        # Determine latest stage by date
        latest_stage_obj = None
        st_ref = getattr(topic_obj, "references", {}).get("hasStages") if hasattr(topic_obj, "references") else None
        if st_ref and st_ref.objects:
            latest_stage_obj = sorted(st_ref.objects, key=lambda o: o.properties.get("date") or "", reverse=True)[0]
        if not latest_stage_obj:
            return []

        # From stage -> aboutReports -> report -> hasEntidades (kind==Person or role contains Journalist)
        stage_loaded = self.client.collections.get("Etapa").query.fetch_objects(
            return_properties=["name", "date"],
            return_references=[
                QueryReference(
                    link_on="aboutReports",
                    return_properties=["title"],
                    return_references=QueryReference(link_on="hasEntidades", return_properties=["name", "kind", "role"]),
                )
            ],
            limit=1,
        )
        # Find the concrete stage by name+date
        target = None
        for st in getattr(stage_loaded, "objects", []) or []:
            if st.properties.get("name") == latest_stage_obj.properties.get("name") and st.properties.get("date") == latest_stage_obj.properties.get("date"):
                target = st
                break
        if not target:
            return []

        reporters: List[str] = []
        rep_ref = getattr(target, "references", {}).get("aboutReports") if hasattr(target, "references") else None
        if rep_ref:
            for rep in rep_ref.objects:
                ents_ref = getattr(rep, "references", {}).get("hasEntidades") if hasattr(rep, "references") else None
                if ents_ref:
                    for ent in ents_ref.objects:
                        kind = (ent.properties.get("kind") or "").lower()
                        role = (ent.properties.get("role") or "").lower()
                        if kind == "person" or "journalist" in role:
                            name = ent.properties.get("name")
                            if name:
                                reporters.append(name)
        return sorted(set(reporters))

    # -------------------------- Hybrid search demo --------------------------
    def hybrid_search_all(self, query_text: str, alpha: float = 0.2, limit: int = 5) -> None:
        """Run a simple hybrid (BM25 + vector) search across all newsroom collections.

        Notes:
        - If the collection lacks vectorization, we fallback to alpha=0.0 (pure BM25).
        - Results are printed sorted by score (desc).
        """
        collections_props = {
            "Entidade": ["name", "kind", "role"],
            "Ficheiro": ["title", "date"],  # body can be long; omit in print
            "Fluxo": ["name", "description"],
            "Etapa": ["name", "date", "category"],
        }

        aggregated = []
        for coll_name, props in collections_props.items():
            coll = self.client.collections.get(coll_name)
            try:
                res = coll.query.hybrid(
                    query=query_text,
                    alpha=alpha,
                    return_properties=props,
                    return_metadata=MetadataQuery(score=True),
                    limit=limit,
                )
            except Exception:
                # Likely missing vectors; fallback to BM25-only
                res = coll.query.hybrid(
                    query=query_text,
                    alpha=0.0,
                    return_properties=props,
                    return_metadata=MetadataQuery(score=True),
                    limit=limit,
                )

            for obj in getattr(res, "objects", []) or []:
                aggregated.append(
                    {
                        "class": coll_name,
                        "props": getattr(obj, "properties", {}) or {},
                        "score": getattr(getattr(obj, "metadata", None), "score", None),
                    }
                )

        aggregated.sort(key=lambda x: (x["score"] is not None, x["score"]), reverse=True)
        print("\n[Hybrid] Results:")
        for r in aggregated:
            main_label = r["props"].get("name") or r["props"].get("title") or r["props"].get("description")
            print(f"[{r['class']}] {main_label} (score={r['score']})")

    # -------------------------- Example run --------------------------
    def demo_insert_and_answer(self) -> None:
        """Insert the real-life example and answer the requested questions via structured queries."""
        print("Resetting schema...")
        self.reset_schema()

        # Entities
        print("Creating entities...")
        russia = self.create_entity("Russia", kind="Country")
        ukraine = self.create_entity("Ukraine", kind="Country")
        journalist = self.create_entity("Jane Doe", kind="Person", role="Journalist")

        # Topic (Fluxo)
        print("Creating topic (fluxo)...")
        topic = self.create_topic("Russia vs Ukraine War", description="Coverage of the ongoing Russia-Ukraine war.")

        # Report (Ficheiro)
        print("Creating report (ficheiro)...")
        report = self.create_report(
            title="Drone attack hits military base",
            body=(
                "A drone strike has reportedly destroyed parts of a military base near the front. "
                "Witnesses describe significant damage and casualties."
            ),
            date_iso="2024-07-12T00:00:00Z",
        )

        # Link report to entities and topic
        print("Linking report to entities and topic...")
        self.link_report_to_entities(report, [russia, ukraine, journalist])
        self.link_report_to_topic(report, topic)

        # Stage (Etapa) caused by the report
        print("Creating and linking stage (etapa)...")
        stage = self.create_stage(
            name="Destruction of a military base",
            date_iso="2024-07-12T00:00:00Z",
            description="Base infrastructure severely damaged following a drone attack.",
            category="war_crime",  # for demo purposes we tag it; adjust as needed
        )
        self.link_stage_to_topic_and_report_and_entities(stage, topic, report, [russia, ukraine])

        print("Insertion complete. Now answering questions...")

        # Q1: How is the war on Russia and Ukraine?
        status = self.topic_status("Russia vs Ukraine War")
        print("\n[Q1] How is the war on Russia and Ukraine?")
        if status.get("exists"):
            latest = status.get("latest_stage")
            print(f"Topic: {status['topic']} | Reports: {status['report_count']}")
            if latest:
                print(f"Latest stage on {latest.get('date')}: {latest.get('name')} ({latest.get('category')})")
                print(f"  Details: {latest.get('description')}")
            else:
                print("No stages recorded yet.")
        else:
            print("Topic not found.")

        # Q2: What is the latest war crime?
        latest_wc = self.latest_war_crime("Russia vs Ukraine War")
        print("\n[Q2] What is the latest war crime?")
        if latest_wc:
            print(f"Latest war crime on {latest_wc.get('date')}: {latest_wc.get('name')}")
            print(f"  Details: {latest_wc.get('description')}")
        else:
            print("No war crimes found for this topic.")

        # Q3: Who reported?
        reporters = self.reporters_for_latest_stage("Russia vs Ukraine War")
        print("\n[Q3] Who reported?")
        if reporters:
            print("Reporters: " + ", ".join(reporters))
        else:
            print("No reporters linked to the latest stage.")

        # Q4: Hybrid search demo
        print("\n[Q4] Hybrid search (query='drone attack')")
        self.hybrid_search_all("drone attack", alpha=0.2, limit=3)

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass


def main() -> None:
    print("Connecting to Weaviate...")
    app = NewsWeaviate(connect_to_local=True)
    print("Connected. Running demo...")
    try:
        app.demo_insert_and_answer()
    finally:
        app.close()
        print("Done.")


if __name__ == "__main__":
    main()
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
        """
        Journalism example: structured schema, a real report, and targeted structured queries.

        This replaces benchmarking with a concrete workflow:
        - Entities (Entidade): Russia, Ukraine, and the Journalist who reported
        - Topic (Fluxo): "Russia vs Ukraine War"
        - Stage/Event (Etapa): "Destruction of a military base" with an ISO date
        - Report (Ficheiro): A report about a drone attack linked to the entities and the topic; the report caused the stage

        Then we answer structured questions like:
        - How is the war on Russia and Ukraine? (retrieve latest stages and counts for the topic)
        - What is the latest war crime? (filter stages by category)
        - Who reported? (traverse from latest stage -> report -> journalist entity)

        We prioritize structured traversal/sorting over semantic search.
        """

        from typing import Any, Dict, List, Optional, Tuple
        import weaviate
        from weaviate.classes.config import DataType, Property, ReferenceProperty
        from weaviate.classes.query import QueryReference


        class NewsWeaviate:
            def __init__(self, *, connect_to_local: bool = True) -> None:
                self.client = weaviate.connect_to_local() if connect_to_local else weaviate.connect_to_local()

            # -------------------------- Schema --------------------------
            def reset_schema(self) -> None:
                """Drop and recreate a minimal schema tailored for a newsroom."""
                for name in ["Etapa", "Ficheiro", "Fluxo", "Entidade"]:
                    try:
                        self.client.collections.delete(name)
                    except Exception:
                        pass

                # Entidade: a named entity (country, person, org) optionally with a role
                self._create_collection(
                    name="Entidade",
                    description="Named entity such as a country, person, or organization.",
                    properties=[
                        Property(name="name", data_type=DataType.TEXT),
                        Property(name="kind", data_type=DataType.TEXT),  # Country | Person | Organization
                        Property(name="role", data_type=DataType.TEXT),  # e.g., Journalist
                    ],
                )

                # Ficheiro: a news report
                self._create_collection(
                    name="Ficheiro",
                    description="News report (file/document) with title, body, and date.",
                    properties=[
                        Property(name="title", data_type=DataType.TEXT),
                        Property(name="body", data_type=DataType.TEXT),
                        Property(name="date", data_type=DataType.DATE),  # ISO-8601 (YYYY-MM-DD)
                    ],
                )

                # Fluxo: a topic/storyline being followed (e.g., Russia vs Ukraine War)
                self._create_collection(
                    name="Fluxo",
                    description="Topic or storyline with multiple stages and reports.",
                    properties=[
                        Property(name="name", data_type=DataType.TEXT),
                        Property(name="description", data_type=DataType.TEXT),
                    ],
                )

                # Etapa: a stage/event in a topic, often caused or evidenced by a report
                self._create_collection(
                    name="Etapa",
                    description="Event/stage in a topic with date and category.",
                    properties=[
                        Property(name="name", data_type=DataType.TEXT),
                        Property(name="date", data_type=DataType.DATE),
                        Property(name="description", data_type=DataType.TEXT),
                        Property(name="category", data_type=DataType.TEXT),  # e.g., war_crime, attack, diplomacy
                    ],
                )

                # Add references
                ent = self.client.collections.get("Entidade")
                ent.config.add_reference(ReferenceProperty(name="hasReports", target_collection="Ficheiro"))

                fich = self.client.collections.get("Ficheiro")
                fich.config.add_reference(ReferenceProperty(name="hasEntidades", target_collection="Entidade"))
                fich.config.add_reference(ReferenceProperty(name="partOfFlux", target_collection="Fluxo"))
                fich.config.add_reference(ReferenceProperty(name="triggersEtapas", target_collection="Etapa"))

                flx = self.client.collections.get("Fluxo")
                flx.config.add_reference(ReferenceProperty(name="hasStages", target_collection="Etapa"))
                flx.config.add_reference(ReferenceProperty(name="hasReports", target_collection="Ficheiro"))

                etp = self.client.collections.get("Etapa")
                etp.config.add_reference(ReferenceProperty(name="belongsToFlux", target_collection="Fluxo"))
                etp.config.add_reference(ReferenceProperty(name="aboutReports", target_collection="Ficheiro"))
                etp.config.add_reference(ReferenceProperty(name="aboutEntities", target_collection="Entidade"))

                print("Schema created: Entidade, Ficheiro, Fluxo, Etapa with references.")

            # -------------------------- Helpers --------------------------
            def _get_by_name(self, coll, name_prop: str, name_val: str):
                """Fetch first object where properties[name_prop] == name_val (simple client-side filter)."""
                res = coll.query.fetch_objects(return_properties=[name_prop], limit=200)
                for obj in getattr(res, "objects", []) or []:
                    if obj.properties.get(name_prop) == name_val:
                        return obj
                return None
            def _create_collection(self, name, description, properties):
                OLLAMA_API_ENDPOINT = "http://host.docker.internal:11434"
                GENERATION_MODEL = "qwen2.5:latest"
                """Create a collection with vectorization & generative AI."""
                properties_list = []
                for prop in properties:
                    properties_list.append(prop.name)
                print(properties_list)
                self.client.collections.create(
                    name=name,
                    description=description,
                    properties=properties,
                vectorizer_config=[
                    Configure.NamedVectors.text2vec_transformers(
                        name="text_vector",
                        pooling_strategy="masked_mean",
                        )
                ],
                    generative_config=Configure.Generative.ollama(
                        api_endpoint=OLLAMA_API_ENDPOINT,
                        model=GENERATION_MODEL
                    )
                )

            # -------------------------- Inserts --------------------------
            def create_entity(self, name: str, kind: str, role: Optional[str] = None):
                return self.client.collections.get("Entidade").data.insert({"name": name, "kind": kind, "role": role or ""})

            def create_topic(self, name: str, description: str = ""):
                return self.client.collections.get("Fluxo").data.insert({"name": name, "description": description})

            def create_report(self, title: str, body: str, date_iso: str):
                return self.client.collections.get("Ficheiro").data.insert({"title": title, "body": body, "date": date_iso})

            def create_stage(self, name: str, date_iso: str, description: str, category: str):
                return self.client.collections.get("Etapa").data.insert({
                    "name": name,
                    "date": date_iso,
                    "description": description,
                    "category": category,
                })

            # -------------------------- Linking --------------------------
            def link_report_to_entities(self, report_obj, entity_objs: List[Any]) -> None:
                ficheiro = self.client.collections.get("Ficheiro")
                entidade = self.client.collections.get("Entidade")
                for ent in entity_objs:
                    try:
                        ficheiro.data.reference_add(report_obj, "hasEntidades", ent)
                    except Exception:
                        pass
                    try:
                        entidade.data.reference_add(ent, "hasReports", report_obj)
                    except Exception:
                        pass

            def link_report_to_topic(self, report_obj, topic_obj) -> None:
                ficheiro = self.client.collections.get("Ficheiro")
                fluxo = self.client.collections.get("Fluxo")
                try:
                    ficheiro.data.reference_add(report_obj, "partOfFlux", topic_obj)
                except Exception:
                    pass
                try:
                    fluxo.data.reference_add(topic_obj, "hasReports", report_obj)
                except Exception:
                    pass

            def link_stage_to_topic_and_report_and_entities(self, stage_obj, topic_obj, report_obj, entities: List[Any]) -> None:
                etapa = self.client.collections.get("Etapa")
                fluxo = self.client.collections.get("Fluxo")
                ficheiro = self.client.collections.get("Ficheiro")
                # Stage -> Flux
                try:
                    etapa.data.reference_add(stage_obj, "belongsToFlux", topic_obj)
                except Exception:
                    pass
                try:
                    fluxo.data.reference_add(topic_obj, "hasStages", stage_obj)
                except Exception:
                    pass

                # Stage <-> Report
                try:
                    etapa.data.reference_add(stage_obj, "aboutReports", report_obj)
                except Exception:
                    pass
                try:
                    ficheiro.data.reference_add(report_obj, "triggersEtapas", stage_obj)
                except Exception:
                    pass

                # Stage -> Entities (optional for richer traversal)
                for ent in entities:
                    try:
                        etapa.data.reference_add(stage_obj, "aboutEntities", ent)
                    except Exception:
                        pass

            # -------------------------- Structured queries --------------------------
            def topic_status(self, topic_name: str) -> Dict[str, Any]:
                """Return latest stages and counts for a topic (Fluxo), using structured traversal and client-side sorting."""
                # Fetch topic with stages and reports
                res = self.client.collections.get("Fluxo").query.fetch_objects(
                    return_properties=["name", "description"],
                    return_references=[
                        QueryReference(link_on="hasStages", return_properties=["name", "date", "category", "description"]),
                        QueryReference(link_on="hasReports", return_properties=["title", "date"]),
                    ],
                    limit=50,
                )

                topic_obj = None
                for obj in getattr(res, "objects", []) or []:
                    if obj.properties.get("name") == topic_name:
                        topic_obj = obj
                        break
                if not topic_obj:
                    return {"topic": topic_name, "exists": False}

                # Gather stages and sort by date desc
                stages = []
                st_ref = getattr(topic_obj, "references", {}).get("hasStages") if hasattr(topic_obj, "references") else None
                if st_ref:
                    for st in st_ref.objects:
                        stages.append({
                            "name": st.properties.get("name"),
                            "date": st.properties.get("date"),
                            "category": st.properties.get("category"),
                            "description": st.properties.get("description"),
                        })
                stages.sort(key=lambda x: x.get("date") or "", reverse=True)

                # Count reports
                report_count = 0
                rp_ref = getattr(topic_obj, "references", {}).get("hasReports") if hasattr(topic_obj, "references") else None
                if rp_ref:
                    report_count = len(rp_ref.objects or [])

                latest = stages[0] if stages else None
                return {
                    "topic": topic_obj.properties.get("name"),
                    "description": topic_obj.properties.get("description"),
                    "exists": True,
                    "report_count": report_count,
                    "latest_stage": latest,
                    "stages": stages[:5],  # top 5 for brevity
                }

            def latest_war_crime(self, topic_name: str) -> Optional[Dict[str, Any]]:
                """Return the latest stage categorized as a war crime under the topic, if any."""
                res = self.client.collections.get("Fluxo").query.fetch_objects(
                    return_properties=["name"],
                    return_references=[QueryReference(link_on="hasStages", return_properties=["name", "date", "category", "description"])],
                    limit=50,
                )
                topic_obj = None
                for obj in getattr(res, "objects", []) or []:
                    if obj.properties.get("name") == topic_name:
                        topic_obj = obj
                        break
                if not topic_obj:
                    return None

                stages = []
                st_ref = getattr(topic_obj, "references", {}).get("hasStages") if hasattr(topic_obj, "references") else None
                if st_ref:
                    for st in st_ref.objects:
                        if (st.properties.get("category") or "").lower() == "war_crime":
                            stages.append({
                                "name": st.properties.get("name"),
                                "date": st.properties.get("date"),
                                "category": st.properties.get("category"),
                                "description": st.properties.get("description"),
                            })
                stages.sort(key=lambda x: x.get("date") or "", reverse=True)
                return stages[0] if stages else None

            def reporters_for_latest_stage(self, topic_name: str) -> List[str]:
                """Find the journalist(s) who reported the report linked to the latest stage in the topic."""
                # Get latest stage for the topic
                res = self.client.collections.get("Fluxo").query.fetch_objects(
                    return_properties=["name"],
                    return_references=[QueryReference(link_on="hasStages", return_properties=["name", "date"],)],
                    limit=50,
                )
                topic_obj = None
                for obj in getattr(res, "objects", []) or []:
                    if obj.properties.get("name") == topic_name:
                        topic_obj = obj
                        break
                if not topic_obj:
                    return []

                # Determine latest stage by date
                latest_stage_obj = None
                st_ref = getattr(topic_obj, "references", {}).get("hasStages") if hasattr(topic_obj, "references") else None
                if st_ref and st_ref.objects:
                    latest_stage_obj = sorted(st_ref.objects, key=lambda o: o.properties.get("date") or "", reverse=True)[0]
                if not latest_stage_obj:
                    return []

                # From stage -> aboutReports -> report -> hasEntidades (kind==Person or role contains Journalist)
                stage_loaded = self.client.collections.get("Etapa").query.fetch_objects(
                    return_properties=["name", "date"],
                    return_references=[
                        QueryReference(
                            link_on="aboutReports",
                            return_properties=["title"],
                            return_references=QueryReference(link_on="hasEntidades", return_properties=["name", "kind", "role"]),
                        )
                    ],
                    limit=1,
                )
                # Find the concrete stage by name+date
                target = None
                for st in getattr(stage_loaded, "objects", []) or []:
                    if st.properties.get("name") == latest_stage_obj.properties.get("name") and st.properties.get("date") == latest_stage_obj.properties.get("date"):
                        target = st
                        break
                if not target:
                    return []

                reporters: List[str] = []
                rep_ref = getattr(target, "references", {}).get("aboutReports") if hasattr(target, "references") else None
                if rep_ref:
                    for rep in rep_ref.objects:
                        ents_ref = getattr(rep, "references", {}).get("hasEntidades") if hasattr(rep, "references") else None
                        if ents_ref:
                            for ent in ents_ref.objects:
                                kind = (ent.properties.get("kind") or "").lower()
                                role = (ent.properties.get("role") or "").lower()
                                if kind == "person" or "journalist" in role:
                                    name = ent.properties.get("name")
                                    if name:
                                        reporters.append(name)
                return sorted(set(reporters))

            # -------------------------- Example run --------------------------
            def demo_insert_and_answer(self) -> None:
                """Insert the real-life example and answer the requested questions via structured queries."""
                print("Resetting schema...")
                self.reset_schema()

                # Entities
                print("Creating entities...")
                russia = self.create_entity("Russia", kind="Country")
                ukraine = self.create_entity("Ukraine", kind="Country")
                journalist = self.create_entity("Jane Doe", kind="Person", role="Journalist")

                # Topic (Fluxo)
                print("Creating topic (fluxo)...")
                topic = self.create_topic("Russia vs Ukraine War", description="Coverage of the ongoing Russia-Ukraine war.")

                # Report (Ficheiro)
                print("Creating report (ficheiro)...")
                report = self.create_report(
                    title="Drone attack hits military base",
                    body=(
                        "A drone strike has reportedly destroyed parts of a military base near the front. "
                        "Witnesses describe significant damage and casualties."
                    ),
                    date_iso="2024-07-12",
                )

                # Link report to entities and topic
                print("Linking report to entities and topic...")
                self.link_report_to_entities(report, [russia, ukraine, journalist])
                self.link_report_to_topic(report, topic)

                # Stage (Etapa) caused by the report
                print("Creating and linking stage (etapa)...")
                stage = self.create_stage(
                    name="Destruction of a military base",
                    date_iso="2024-07-12",
                    description="Base infrastructure severely damaged following a drone attack.",
                    category="war_crime",  # for demo purposes we tag it; adjust as needed
                )
                self.link_stage_to_topic_and_report_and_entities(stage, topic, report, [russia, ukraine])

                print("Insertion complete. Now answering questions...")

                # Q1: How is the war on Russia and Ukraine?
                status = self.topic_status("Russia vs Ukraine War")
                print("\n[Q1] How is the war on Russia and Ukraine?")
                if status.get("exists"):
                    latest = status.get("latest_stage")
                    print(f"Topic: {status['topic']} | Reports: {status['report_count']}")
                    if latest:
                        print(f"Latest stage on {latest.get('date')}: {latest.get('name')} ({latest.get('category')})")
                        print(f"  Details: {latest.get('description')}")
                    else:
                        print("No stages recorded yet.")
                else:
                    print("Topic not found.")

                # Q2: What is the latest war crime?
                latest_wc = self.latest_war_crime("Russia vs Ukraine War")
                print("\n[Q2] What is the latest war crime?")
                if latest_wc:
                    print(f"Latest war crime on {latest_wc.get('date')}: {latest_wc.get('name')}")
                    print(f"  Details: {latest_wc.get('description')}")
                else:
                    print("No war crimes found for this topic.")

                # Q3: Who reported?
                reporters = self.reporters_for_latest_stage("Russia vs Ukraine War")
                print("\n[Q3] Who reported?")
                if reporters:
                    print("Reporters: " + ", ".join(reporters))
                else:
                    print("No reporters linked to the latest stage.")

            def close(self) -> None:
                try:
                    self.client.close()
                except Exception:
                    pass


        def main() -> None:
            app = NewsWeaviate(connect_to_local=True)
            print("Connected to Weaviate.")
            try:
                app.demo_insert_and_answer()
            finally:
                app.close()


        if __name__ == "__main__":
            main()