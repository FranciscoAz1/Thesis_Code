import weaviate, { WeaviateClient, ApiKey } from 'weaviate-ts-client';
import { Config } from './config.js';
import { Logger } from './logger.js';

export interface WeaviateObject {
  id?: string;
  class?: string;
  properties?: Record<string, any>;
}

export interface QueryResult {
  data?: {
    Get?: Record<string, any[]>;
  };
}

// Options to include a cross-reference in a GraphQL Get query
export interface ReferenceQueryOptions {
  // The name of the reference property on the base class (e.g., "hasEtapas")
  linkOn: string;
  // The target class name of the referenced objects (e.g., "Etapa")
  refClassName: string;
  // Properties to return for the referenced objects (e.g., ["name"])
  refProps: string[];
}

export class WeaviateConnection {
  private client: WeaviateClient;
  private logger: Logger;

  constructor(config: Config, logger: Logger) {
    this.logger = logger;
    
    logger.info(`Connecting to Weaviate at ${config.weaviateScheme}://${config.weaviateHost}`);
    
    this.client = weaviate.client({
      scheme: config.weaviateScheme,
      host: config.weaviateHost,
      // Add authentication if needed
      // apiKey: new ApiKey('your-api-key'),
    });

    logger.info('Successfully connected to Weaviate');
  }

  async insertOne(collection: string, properties: Record<string, any>): Promise<WeaviateObject> {
    try {
      const obj: WeaviateObject = {
        class: collection,
        properties,
      };

      const result = await this.client.data
        .creator()
        .withClassName(collection)
        .withProperties(properties)
        .do();

      return {
        id: result.id,
        class: collection,
        properties,
      };
    } catch (error) {
      throw new Error(`Failed to insert object: ${error}`);
    }
  }

  async query(
    collection: string,
    query: string,
    targetProps: string[],
    limit: number = 3
  ): Promise<string> {
    try {
      const schema = await this.getSchema().catch(() => undefined);
      // const useHybrid = this.hasVectorizer(schema, collection);
      const useHybrid = true;

      let queryBuilder = this.client.graphql
        .get()
        .withClassName(collection)
        .withFields(targetProps.join(' '));
      if (useHybrid) {
        queryBuilder = queryBuilder.withHybrid({ query: query });
      } else {
        this.logger.warn(`Class ${collection} has no vectorizer configured; hybrid search skipped.`);
      }

      if (limit > 0) {
        queryBuilder = queryBuilder.withLimit(limit);
      }

      const result = await queryBuilder.do();
      const json = JSON.stringify(result, null, 2);
      if (!useHybrid) return `Warning: class ${collection} has no vectorizer configured; hybrid search skipped.\n${json}`;
      return json;
    } catch (error) {
      throw new Error(`Query failed: ${error}`);
    }
  }



  async generateText(
    collection: string,
    query: string,
    targetProps: string[],
    limit: number = 3,
  ): Promise<string> {
    try {
      const generativePrompt = `Answer briefly: ${query}`;
      const schema = await this.getSchema().catch(() => undefined);
      const useHybrid = this.hasVectorizer(schema, collection);

      let queryBuilder = this.client.graphql
        .get()
        .withClassName(collection)
        .withFields(targetProps.join(' '))
        .withGenerate({ groupedTask: generativePrompt, groupedProperties: targetProps });

      if (useHybrid) {
        queryBuilder = queryBuilder.withHybrid({ query: query });
      } else {
        this.logger.warn(`Class ${collection} has no vectorizer configured; hybrid generation skipped.`);
      }
      if (limit > 0) queryBuilder = queryBuilder.withLimit(limit);
      const result = await queryBuilder.do();
      const json = JSON.stringify(result, null, 2);
      if (!useHybrid) return `Warning: class ${collection} has no vectorizer configured; hybrid generation skipped.\n${json}`;
      return json;
    } catch (error) {
      throw new Error(`Text generation failed: ${error}`);
    }
  }

  async getClassSchema(className: string): Promise<any> {
    try {
      const schema = await this.client.schema
        .classGetter()
        .withClassName(className)
        .do();
      
      return schema;
    } catch (error) {
      throw new Error(`Failed to get class schema: ${error}`);
    }
  }

  async getSchema(): Promise<any> {
    try {
      const schema = await this.client.schema.getter().do();
      return schema;
    } catch (error) {
      throw new Error(`Failed to get schema: ${error}`);
    }
  }

  // --- Schema utilities for discovering cross references dynamically ---
  private refsForClass(schema: any, className: string): Array<{ prop: string; targets: string[] }> {
    try {
      const cls = schema?.classes?.find((c: any) => c.class === className);
      const props: Array<any> = cls?.properties || [];
      // Consider datatypes that look like class names (capitalized)
      return props
        .map((p: any) => ({
          prop: p.name,
          targets: (p.dataType || []).filter((dt: string) => /^[A-Z]/.test(dt)),
        }))
        .filter((x) => x.targets.length > 0);
    } catch {
      return [];
    }
  }

  private incomingRefsForClass(schema: any, targetClass: string): Array<{ fromClass: string; prop: string }> {
    const incoming: Array<{ fromClass: string; prop: string }> = [];
    try {
      for (const c of schema?.classes || []) {
        for (const p of c?.properties || []) {
          const targets: string[] = (p.dataType || []).filter((dt: string) => /^[A-Z]/.test(dt));
          if (targets.includes(targetClass)) {
            incoming.push({ fromClass: c.class, prop: p.name });
          }
        }
      }
    } catch {
      // ignore
    }
    return incoming;
  }

  private refTargetsForProp(schema: any, className: string, propName: string): string[] {
    try {
      const cls = schema?.classes?.find((c: any) => c.class === className);
      const prop = (cls?.properties || []).find((p: any) => p.name === propName);
      const targets: string[] = (prop?.dataType || []).filter((dt: string) => /^[A-Z]/.test(dt));
      return targets;
    } catch {
      return [];
    }
  }

  // Very small heuristic to determine whether a class has a vectorizer configured.
  // We treat a class as vectorized if it declares a non-'none' `vectorizer` or
  // has a non-empty moduleConfig. This is conservative but avoids calling
  // hybrid on classes that would error.
  private hasVectorizer(schema: any, className: string): boolean {
    try {
      const cls = schema?.classes?.find((c: any) => c.class?.toLowerCase() === className?.toLowerCase());
      if (!cls) return false;
      if (typeof cls.vectorizer === 'string' && cls.vectorizer.trim() !== '' && cls.vectorizer.trim().toLowerCase() !== 'none') {
        return true;
      }
      if (cls.moduleConfig && Object.keys(cls.moduleConfig).length > 0) return true;
      return false;
    } catch {
      return false;
    }
  }

  /**
   * Hybrid search Etapa objects and return cross-references to Fluxo and Entidade.
   * - Searches a collection with hybrid query, and returns possible cross-references.
   */
  async queryOrigin(
    collection: string,
    query: string,
    limit: number = 5,
    targetProps: string[] = ['name']
  ): Promise<string> {
    try {

      const schema = await this.getSchema().catch(() => undefined);
      const useHybrid = this.hasVectorizer(schema, collection);

      let qb = this.client.graphql
        .get()
        .withClassName(collection)
        .withFields(targetProps.join(' '));

      if (useHybrid) {
        qb = qb.withHybrid({ query });
      } else {
        this.logger.warn(`Class ${collection} has no vectorizer configured; hybrid search skipped.`);
      }

      if (limit > 0) qb = qb.withLimit(limit);

      const result = await qb.do();

      // Parse results into a simplified structure
      const rows: Array<any> = result?.data?.Get?.Etapa || [];
      type SimpleRow = { etapa?: string; fluxo?: string | null; entidades: string[] };
      const simplified: SimpleRow[] = rows.map((r: any) => {
        const etapa = r?.name;
        const fluxo = (r?.belongsToFluxo || [])[0]?.name ?? null;
        const entidades: string[] = [];
        for (const fich of r?.hasFicheiros || []) {
          for (const ent of fich?.hasEntidades || []) {
            if (ent?.name) entidades.push(ent.name);
          }
        }
        return { etapa, fluxo, entidades: Array.from(new Set(entidades)) };
      });

  // Discover available collections and reference paths to suggest deepening
  const classes: string[] = schema?.classes?.map((c: any) => c.class) ?? [];

  // Determine properties present on the collection so we can report any
  // fields that weren't requested (missed fields) to help the LLM explore.
  const classProps: string[] = (schema?.classes?.find((c: any) => c.class === collection)?.properties || []).map((p: any) => p.name);
  const requested = new Set(targetProps.map((p) => p));
  const missedFields = classProps.filter((p) => !requested.has(p));

      const refsFor = (className: string): Array<{ prop: string; targets: string[] }> => {
        try {
          const cls = schema?.classes?.find((c: any) => c.class === className);
          const props: Array<any> = cls?.properties || [];
          return props
            .map((p: any) => ({ prop: p.name, targets: (p.dataType || []).filter((dt: string) => /^[A-Z]/.test(dt)) }))
            .filter((x) => x.targets.length > 0);
        } catch {
          return [];
        }
      };

      // Dynamic discovery of references starting from Etapa
      const etapaOutgoing = this.refsForClass(schema, collection);
      const etapaIncoming = this.incomingRefsForClass(schema, collection);

      // Gather first-degree neighbor classes from Etapa's outgoing and incoming edges
      const neighborClasses = new Set<string>();
      etapaOutgoing.forEach((r) => r.targets.forEach((t) => neighborClasses.add(t)));
      etapaIncoming.forEach((r) => neighborClasses.add(r.fromClass));
      neighborClasses.delete(collection);

      // Build a map of neighbor -> outgoing refs (to suggest deepening)
      const neighborOutgoing: Record<string, Array<{ prop: string; targets: string[] }>> = {};
      for (const nc of neighborClasses) {
        neighborOutgoing[nc] = this.refsForClass(schema, nc);
      }

      // Build a natural-language response suitable for an LLM
      const lines: string[] = [];
      lines.push(`Hybrid search results for Etapa (query="${query}", limit=${limit}).`);
      if (!useHybrid) {
        lines.push(`Warning: class ${collection} has no vectorizer configured; hybrid search was skipped.`);
      }
      if (classes.length) {
        lines.push(`Collections available in the schema: ${classes.join(', ')}.`);
      }
      if (simplified.length === 0) {
        lines.push('No Etapa matched your query.');
      } else {
        lines.push('Matched Etapas with their linked Fluxo and Entidade(s):');
        for (const item of simplified) {
          const entTxt = item.entidades.length ? item.entidades.join(', ') : 'none';
          lines.push(`- Etapa: ${item.etapa} | Fluxo: ${item.fluxo ?? 'none'} | Entidades (via Ficheiro): ${entTxt}`);
        }
      }

      // Mention any fields present in the collection that were not requested
      // so the LLM can consider inspecting them as well.
      if (missedFields.length) {
        // Limit the list length to keep responses concise
        const show = missedFields.slice(0, 12);
        lines.push(`Missed fields on ${collection}: ${show.join(', ')}${missedFields.length > show.length ? ', ...' : ''}`);
      }

      // Describe dynamic connections
      const describeOutgoing = (title: string, refs: Array<{ prop: string; targets: string[] }>) => {
        if (!refs.length) return;
        lines.push(`${title}:`);
        for (const r of refs) lines.push(`- ${r.prop} -> ${r.targets.join(' | ')}`);
      };

      const describeIncoming = (title: string, refs: Array<{ fromClass: string; prop: string }>) => {
        if (!refs.length) return;
        lines.push(`${title}:`);
        for (const r of refs) lines.push(`- ${r.fromClass}.${r.prop} -> Etapa`);
      };

      lines.push('You can deepen the search by following these connections:');
      describeOutgoing('From Etapa (outgoing refs)', etapaOutgoing);
      describeIncoming('To Etapa (incoming refs)', etapaIncoming);
      // if (neighborClasses.size) {
      //   lines.push('Neighbor classes and their outgoing refs:');
      //   for (const nc of Array.from(neighborClasses).sort()) {
      //     describeOutgoing(`- From ${nc}`, neighborOutgoing[nc] || []);
      //   }
      // }
      return lines.join('\n');
    } catch (error) {
      throw new Error(`Query failed: ${error}`);
    }
  }

  // Dynamic, one-hop traversal on a chosen reference prop
  async queryWithRefs(
    collection: string,
    refProp: string,
    query: string,
    limit: number = 5,
    baseProps: string[] = ['name'],
    refProps: string[] = ['name']
  ): Promise<string> {
    try {
      const schema = await this.getSchema().catch(() => undefined);
      const targets = this.refTargetsForProp(schema, collection, refProp);

      // Build fields: base props + reference fragments for each possible target class
      const baseFields = baseProps.join(' ');
      let refBlock = '';
      if (targets.length > 0) {
        const fragments = targets.map((t) => `... on ${t} { ${refProps.join(' ')} }`).join(' ');
        refBlock = ` ${refProp} { ${fragments} }`;
      } else {
        // If we can't resolve targets, still request the ref block without fragments (best-effort)
        refBlock = ` ${refProp} { ${refProps.join(' ')} }`;
      }
      const fields = `${baseFields}${refBlock}`.trim();

      const useHybrid = this.hasVectorizer(schema, collection);

      let qb = this.client.graphql
        .get()
        .withClassName(collection)
        .withFields(fields);
      if (useHybrid) {
        qb = qb.withHybrid({ query });
      } else {
        this.logger.warn(`Class ${collection} has no vectorizer configured; hybrid search skipped.`);
      }
      if (limit > 0) qb = qb.withLimit(limit);
      
      const result = await qb.do();
      const rows: Array<any> = result?.data?.Get?.[collection] || [];
      const toName = (o: any) => (o && typeof o === 'object' ? o.name ?? JSON.stringify(o) : String(o));

      type RowSummary = { base: string; refs: string[] };
      const summarized: RowSummary[] = rows.map((r: any) => {
        const base = r?.name ?? toName(r);
        const refsArr: any[] = Array.isArray(r?.[refProp]) ? r[refProp] : [];
        const refs = refsArr.map(toName).filter(Boolean);
        return { base, refs };
      });

      // Also provide dynamic connections for the base class to help choose next hops
      const classes: string[] = schema?.classes?.map((c: any) => c.class) ?? [];
      const outgoing = this.refsForClass(schema, collection);
      const incoming = this.incomingRefsForClass(schema, collection);
      const neighborClasses = new Set<string>();
      outgoing.forEach((r) => r.targets.forEach((t) => neighborClasses.add(t)));
      incoming.forEach((r) => neighborClasses.add(r.fromClass));
      neighborClasses.delete(collection);
      const neighborOutgoing: Record<string, Array<{ prop: string; targets: string[] }>> = {};
      for (const nc of neighborClasses) neighborOutgoing[nc] = this.refsForClass(schema, nc);

      const lines: string[] = [];
      lines.push(`Hybrid search for ${collection} following reference '${refProp}' (query="${query}", limit=${limit}).`);
      if (!useHybrid) {
        lines.push(`Warning: class ${collection} has no vectorizer configured; hybrid search was skipped.`);
      }
      if (classes.length) lines.push(`Collections available: ${classes.join(', ')}.`);

      // Suggest any fields from the base collection that were not requested
      // (so the LLM can consider inspecting them too).
      const classProps: string[] = (schema?.classes?.find((c: any) => c.class === collection)?.properties || []).map((p: any) => p.name);
      const requestedBase = new Set(baseProps.map((p) => p));
      const missedBaseFields = classProps.filter((p) => !requestedBase.has(p));
      if (missedBaseFields.length) {
        const show = missedBaseFields.slice(0, 12);
        lines.push(`Missed fields on ${collection}: ${show.join(', ')}${missedBaseFields.length > show.length ? ', ...' : ''}`);
      }

      // Suggest any fields on the referenced target classes that were not
      // requested in refProps (grouped per target class when possible).
      if (targets.length && schema?.classes) {
        for (const t of targets) {
          const tProps: string[] = (schema.classes.find((c: any) => c.class === t)?.properties || []).map((p: any) => p.name);
          const requestedRef = new Set(refProps.map((p) => p));
          const missedRef = tProps.filter((p) => !requestedRef.has(p));
          if (missedRef.length) {
            const show = missedRef.slice(0, 10);
            lines.push(`Missed fields on ${t}: ${show.join(', ')}${missedRef.length > show.length ? ', ...' : ''}`);
          }
        }
      }

      if (summarized.length === 0) {
        lines.push(`No ${collection} matched your query or no '${refProp}' references found.`);
      } else {
        const targetLabel = targets.length ? targets.join(' | ') : 'UnknownTarget';
        lines.push(`Traversed '${refProp}' -> ${targetLabel}:`);
        for (const item of summarized) {
          const refsText = item.refs.length ? item.refs.join(', ') : 'none';
          lines.push(`- ${collection}: ${item.base} | ${refProp}: ${refsText}`);
        }
      }

      const describeOutgoing = (title: string, refs: Array<{ prop: string; targets: string[] }>) => {
        if (!refs.length) return;
        lines.push(`${title}:`);
        for (const r of refs) lines.push(`- ${r.prop} -> ${r.targets.join(' | ')}`);
      };
      const describeIncoming = (title: string, refs: Array<{ fromClass: string; prop: string }>) => {
        if (!refs.length) return;
        lines.push(`${title}:`);
        for (const r of refs) lines.push(`- ${r.fromClass}.${r.prop} -> ${collection}`);
      };

      lines.push('You can deepen the search by following these connections:');
      describeOutgoing(`From ${collection} (outgoing refs)`, outgoing);
      describeIncoming(`To ${collection} (incoming refs)`, incoming);


      return lines.join('\n');
    } catch (error) {
      throw new Error(`Query with refs failed: ${error}`);
    }
  }
}