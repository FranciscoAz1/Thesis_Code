import express, { Express } from 'express';
import { randomUUID } from 'node:crypto';
import { Server as HttpServer } from 'node:http';
import { McpServer, ResourceTemplate } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { StreamableHTTPServerTransport } from '@modelcontextprotocol/sdk/server/streamableHttp.js';
import { z } from 'zod';

import { Config } from './config.js';
import { Logger } from './logger.js';
import { WeaviateConnection } from './weaviate.js';

const QUERY_TOOL_NAME = 'weaviate-query';
const ORIGIN_TOOL_NAME = 'weaviate-origin';
const FOLLOW_REF_TOOL_NAME = 'weaviate-follow-ref';
const GENERATE_TOOL_NAME = 'weaviate-generate-text';
const SCHEMA_TEMPLATE_NAME = 'weaviate-schema';
const SCHEMA_URI_TEMPLATE = 'weaviate://schema/{collection}';
const COLLECTION_CACHE_TTL_MS = 60_000;

const QUERY_INPUT_SHAPE = {
  query: z.string().min(1, 'query is required'),
  collection: z
    .string()
    .min(1, 'collection cannot be empty'),
  targetProperties: z
    .array(z.string().min(1))
    .min(1, 'targetProperties requires at least one property'),
  limit: z
    .number()
    .int()
    .positive()
    .max(50, 'limit must be <= 50')
    .optional()
} satisfies z.ZodRawShape;

const ORIGIN_INPUT_SHAPE = {
  query: z.string().min(1, 'query is required'),
  collection: z.string().min(1, 'collection cannot be empty'),
  targetProperties: z
    .array(z.string().min(1))
    .min(1, 'targetProperties requires at least one property'),
  limit: z
    .number()
    .int()
    .positive()
    .max(50, 'limit must be <= 50')
    .optional()
} satisfies z.ZodRawShape;

const FOLLOW_REF_INPUT_SHAPE = {
  query: z.string().min(1, 'query is required'),
  collection: z.string().min(1, 'collection cannot be empty'),
  baseProps: z.array(z.string().min(1)).min(1, 'baseProps requires at least one property'),
  refProp: z.string().min(1, 'refProp is required'),
  refProps: z.array(z.string().min(1)).min(1, 'refProps requires at least one property'),
  limit: z.number().int().positive().max(50).optional()
} satisfies z.ZodRawShape;
const GENERATE_INPUT_SHAPE = {
  query: z.string().min(1, 'query is required'),
  collection: z
    .string()
    .min(1, 'collection cannot be empty'),
  targetProperties: z
    .array(z.string().min(1))
    .min(1, 'targetProperties requires at least one property'),
  limit: z
    .number()
    .int()
    .positive()
    .max(50, 'limit must be <= 50')
    .optional()
} satisfies z.ZodRawShape;


const QUERY_INPUT_SCHEMA = z.object(QUERY_INPUT_SHAPE);
const GENERATE_INPUT_SCHEMA = z.object(GENERATE_INPUT_SHAPE);

type QueryInput = z.infer<typeof QUERY_INPUT_SCHEMA>;
type GenerateInput = z.infer<typeof GENERATE_INPUT_SCHEMA>;

interface TransportWrapper {
  close: () => Promise<void>;
}

export class WeaviateMcpRuntime {
  private readonly config: Config;
  private readonly logger: Logger;
  private readonly weaviate: WeaviateConnection;
  private readonly server: McpServer;

  private activeTransport?: TransportWrapper;
  private httpTransport?: StreamableHTTPServerTransport;
  private httpApp?: Express;
  private httpServer?: HttpServer;
  private collectionCache?: { names: string[]; fetchedAt: number };
  private isShuttingDown = false;
  private activeConnections = 0;

  constructor(config: Config, logger: Logger, version = '0.1.0') {
    this.config = config;
    this.logger = logger;

    this.logger.info('Initializing Weaviate connection...');
    this.weaviate = new WeaviateConnection(config, logger);

    this.server = new McpServer(
      {
        name: 'weaviate-mcp-server',
        version
      },
      {
        capabilities: {
          logging: {},
          prompts: { listChanged: true },
          tools: { listChanged: true },
          resources: { listChanged: true }
        }
      }
    );

    this.server.server.onerror = (error) => {
      this.logger.error('Transport error: %s', error instanceof Error ? error.message : String(error));
    };

    this.server.server.onclose = () => {
      this.logger.info('Transport closed');
      // For stdio mode: exit non-zero to let dev supervisors restart on client disconnect.
      // For HTTP mode: the HTTP server remains healthy; don't exit here.
      setImmediate(() => {
        const isHttpMode = Boolean(this.httpServer);
        if (!this.isShuttingDown && !isHttpMode) {
          process.exitCode = 1;
          // Give logger a tick to flush
          setTimeout(() => process.exit(1), 10);
        }
      });
    };

    this.registerTools();
    this.registerResources();

    this.logger.info('MCP Server initialized successfully');
  }

  get instance(): McpServer {
    return this.server;
  }

  async startStdio(): Promise<void> {
    const transport = new StdioServerTransport();
    transport.onerror = (error) => {
      this.logger.error('STDIO transport error: %s', error instanceof Error ? error.message : String(error));
    };
    transport.onclose = () => {
      this.logger.info('STDIO transport closed');
      // Exit to allow dev supervisor to restart on client disconnect
      setImmediate(() => {
        if (!this.isShuttingDown) {
          process.exitCode = 1;
          setTimeout(() => process.exit(1), 10);
        }
      });
    };

    await this.server.connect(transport);
    this.logger.info('STDIO transport connected');

    this.activeTransport = {
      close: async () => {
        await transport.close();
      }
    };
  }

  async startHttp(host: string, port: number): Promise<void> {
    if (!this.httpTransport) {
      this.httpTransport = new StreamableHTTPServerTransport({
        sessionIdGenerator: () => randomUUID(),
        enableJsonResponse: true
      });

      this.httpTransport.onerror = (error) => {
        this.logger.error('HTTP transport error: %s', error instanceof Error ? error.message : String(error));
      };
      this.httpTransport.onclose = () => {
        // Individual HTTP transport/session closures can occur; log and continue.
        this.logger.warn('HTTP transport closed');
      };

      await this.server.connect(this.httpTransport);
      this.logger.info('HTTP transport connected');

      this.activeTransport = {
        close: async () => {
          await this.httpTransport?.close();
        }
      };
    }

    if (!this.httpApp) {
      this.httpApp = express();
      this.httpApp.use(express.json({ limit: '4mb' }));

      const handler = async (req: express.Request, res: express.Response) => {
        const reqId = randomUUID();
        const startedAt = Date.now();
        this.logger.debug(
          'HTTP request start id=%s method=%s url=%s content-length=%s',
          reqId,
          req.method,
          req.url,
          req.headers['content-length'] ?? 'unknown'
        );

        res.on('finish', () => {
          const dur = Date.now() - startedAt;
          this.logger.debug(
            'HTTP response finish id=%s status=%d durationMs=%d',
            reqId,
            res.statusCode,
            dur
          );
        });

        res.on('close', () => {
          const dur = Date.now() - startedAt;
          this.logger.debug('HTTP response close id=%s durationMs=%d', reqId, dur);
        });

        req.on('aborted', () => {
          const dur = Date.now() - startedAt;
          this.logger.warn('HTTP request aborted by client id=%s durationMs=%d', reqId, dur);
        });

        if (!this.httpTransport) {
          res.status(503).json({
            jsonrpc: '2.0',
            error: { code: -32603, message: 'HTTP transport not ready' },
            id: null
          });
          return;
        }

        try {
          await this.httpTransport.handleRequest(req, res, req.body);
        } catch (error) {
          this.logger.error('HTTP request handling failed: %s', error instanceof Error ? error.message : String(error));
          if (!res.headersSent) {
            res.status(500).json({
              jsonrpc: '2.0',
              error: { code: -32603, message: 'Internal server error' },
              id: null
            });
          }
        }
      };

      this.httpApp.post('/mcp', handler);
      this.httpApp.get('/mcp', handler);
      this.httpApp.delete('/mcp', handler);
    }

    await new Promise<void>((resolve, reject) => {
      const server = this.httpApp!.listen(port, host, () => {
        this.logger.info(`HTTP MCP server listening on http://${host}:${port}/mcp`);
        resolve();
      });

      // Tune server timeouts to reduce premature closes under load or slow clients
      // Keep-Alive a bit over 60s, headers a bit longer, disable per-request timeout
      server.keepAliveTimeout = 65_000; // default can be ~5s; extend to keep connections alive longer
      server.headersTimeout = 66_000;   // must be > keepAliveTimeout
  // 0 disables (no automatic timeout). If you prefer a limit, set e.g., 120_000
  (server as unknown as { requestTimeout?: number }).requestTimeout = 0;

      server.on('clientError', (err) => {
        this.logger.warn('HTTP clientError: %s', err instanceof Error ? err.message : String(err));
      });

      server.on('connection', (socket) => {
        this.activeConnections += 1;
        this.logger.debug('HTTP connection opened, active=%d', this.activeConnections);
        socket.on('close', () => {
          this.activeConnections = Math.max(0, this.activeConnections - 1);
          this.logger.debug('HTTP connection closed, active=%d', this.activeConnections);
        });
      });

      server.on('error', (error: unknown) => {
        this.logger.error('HTTP server error: %s', error instanceof Error ? error.message : String(error));
        reject(error instanceof Error ? error : new Error(String(error)));
      });

      // If the HTTP server closes unexpectedly, exit non-zero so a dev supervisor (e.g., nodemon)
      // can restart the process. During an intentional shutdown we set isShuttingDown to true
      // to avoid triggering a restart.
      server.on('close', () => {
        this.logger.info('HTTP server closed');
        if (!this.isShuttingDown) {
          setImmediate(() => {
            process.exitCode = 1;
            setTimeout(() => process.exit(1), 10);
          });
        }
      });

      this.httpServer = server;
    });
  }

  async shutdown(): Promise<void> {
    this.isShuttingDown = true;
    if (this.httpServer) {
      await new Promise<void>((resolve) => {
        this.httpServer?.close(() => resolve());
      });
      this.logger.info('HTTP server closed');
      this.httpServer = undefined;
    }

    if (this.activeTransport) {
      await this.activeTransport.close();
      this.activeTransport = undefined;
    }

    await this.server.close();
    this.logger.info('MCP server closed');
  }

  private registerTools(): void {
    if (this.isToolDisabled(QUERY_TOOL_NAME)) {
      this.logger.info('Tool %s disabled via configuration', QUERY_TOOL_NAME);
    } else {
      this.server.registerTool(
        QUERY_TOOL_NAME,
        {
          title: 'Weaviate Hybrid Query',
          description: 'Query objects from a Weaviate collection using hybrid search',
          inputSchema: QUERY_INPUT_SHAPE
        },
        async ({ query, collection, targetProperties, limit }: QueryInput) => {
          const { name: targetCollection, schema } = await this.resolveCollection(collection);
          const validatedProperties = await this.validateTargetProperties(targetCollection, targetProperties, schema);
          const effectiveLimit = limit ?? 3;

          const rawResult = await this.weaviate.query(targetCollection, query, validatedProperties, effectiveLimit);
          const structured = this.safeParseJson(rawResult);
          const structuredContent = this.normalizeStructuredContent(structured);

          this.logger.info('Query success: result bytes=%d', rawResult.length);

          const response: {
            content: Array<{ type: 'text'; text: string }>;
            structuredContent?: Record<string, unknown>;
          } = {
            content: [
              {
                type: 'text',
                text: structuredContent ? JSON.stringify(structuredContent, null, 2) : rawResult
              }
            ]
          };

          if (structuredContent) {
            response.structuredContent = structuredContent;
          }

          return response;
        }
      );
    }
    // Add other two types of queries here, with checks like in weaviate-query, query-tool-name

    // 1) Origin-style natural-language summary starting from a collection
    if (this.isToolDisabled(ORIGIN_TOOL_NAME)) {
      this.logger.info('Tool %s disabled via configuration', ORIGIN_TOOL_NAME);
    } else {
      type OriginInput = z.infer<typeof ORIGIN_INPUT_SCHEMA>;
      const ORIGIN_INPUT_SCHEMA = z.object(ORIGIN_INPUT_SHAPE);
      this.server.registerTool(
        ORIGIN_TOOL_NAME,
        {
          title: 'Weaviate Hybrid Query with cross-references',
          description: 'Query objects from a Weaviate collection using hybrid search and get next-hop connections',
          inputSchema: ORIGIN_INPUT_SHAPE
        },
        async ({ query, collection, targetProperties, limit }: OriginInput) => {
          const { name: targetCollection, schema } = await this.resolveCollection(collection);
          const validatedProperties = await this.validateTargetProperties(targetCollection, targetProperties, schema);
          const effectiveLimit = limit ?? 5;

          // For now queryOrigin expects Etapa-style fields; it's safe when used on Etapa
          // If used on other classes, it still returns collection info and connections.
          const nl = await this.weaviate.queryOrigin(targetCollection, query, effectiveLimit, ['name']);
          return { content: [{ type: 'text', text: nl }] };
        }
      );
    }

    // 2) Follow a specific reference from a collection (one-hop traversal)
    if (this.isToolDisabled(FOLLOW_REF_TOOL_NAME)) {
      this.logger.info('Tool %s disabled via configuration', FOLLOW_REF_TOOL_NAME);
    } else {

      type FollowRefInput = z.infer<typeof FOLLOW_REF_INPUT_SCHEMA>;
      const FOLLOW_REF_INPUT_SCHEMA = z.object(FOLLOW_REF_INPUT_SHAPE);

      this.server.registerTool(
        FOLLOW_REF_TOOL_NAME,
        {
          title: 'Weaviate Follow Reference',
          description: 'Query objects from a Weaviate collection using hybrid search and get next-hop connections by following a specific reference property, from a previous next-hop suggestion',
          inputSchema: FOLLOW_REF_INPUT_SHAPE
        },
        async ({ query, collection, refProp, baseProps, refProps, limit }: FollowRefInput) => {
          const { name: targetCollection, schema } = await this.resolveCollection(collection);
          const classSchema = schema ?? (await this.weaviate.getClassSchema(targetCollection));

          // Validate baseProps (exist in base class)
          const effectiveBaseProps = baseProps && baseProps.length ? baseProps : ['name'];
          await this.validateTargetProperties(targetCollection, effectiveBaseProps, classSchema);

          // Validate that refProp exists and points to at least one target class
          const rawProps: Array<any> = classSchema?.properties || [];
          const ref = rawProps.find((p) => p?.name === refProp);
          if (!ref) {
            // Only surface properties that look like references (dataType contains a capitalized class name)
            const refPropsOnly = rawProps
              .filter((p) => Array.isArray(p?.dataType) && (p.dataType || []).some((dt: string) => /^[A-Z]/.test(dt)))
              .map((p) => p.name);
            const allowed = refPropsOnly.length > 0 ? refPropsOnly.join(', ') : rawProps.map((p) => p.name).join(', ');
            throw new Error(`Reference property '${refProp}' does not exist on '${targetCollection}'. Available reference properties: ${allowed}`);
          }
          const targets: string[] = (ref.dataType || []).filter((dt: string) => /^[A-Z]/.test(dt));
          if (!targets.length) {
            throw new Error(`Property '${refProp}' on '${targetCollection}' does not appear to be a cross-reference (no target class in dataType).`);
          }

          // Validate refProps against target class(es) if provided
          const effectiveRefProps = refProps && refProps.length ? refProps : ['name'];
          try {
            // Validate against the first target (common case); if differing schemas exist, skip strict validation
            const firstTarget = targets[0];
            const targetSchema = await this.weaviate.getClassSchema(firstTarget);
            const allowedTargetProps = new Set((targetSchema?.properties || []).map((p: any) => p.name));
            for (const p of effectiveRefProps) {
              if (!allowedTargetProps.has(p)) {
                throw new Error(`Ref property '${p}' not found on target class '${firstTarget}'. Available: ${Array.from(allowedTargetProps).join(', ')}`);
              }
            }
          } catch (e) {
            // If schema lookup fails, proceed without validation
            this.logger.warn('Skipping strict refProps validation: %s', e instanceof Error ? e.message : String(e));
          }

          const effectiveLimit = limit ?? 5;
            const nl = await this.weaviate.queryWithRefs(
              targetCollection,
              refProp,
              query,
              effectiveLimit,
              effectiveBaseProps,
              effectiveRefProps
            );
            return { content: [{ type: 'text', text: nl }] };
        }
      );
    }

    if (this.isToolDisabled(GENERATE_TOOL_NAME)) {
      this.logger.info('Tool %s disabled via configuration', GENERATE_TOOL_NAME);
    } else {
      this.server.registerTool(
        GENERATE_TOOL_NAME,
        {
          title: 'Weaviate Generate Text',
          description: 'Generate text using Weaviate\'s generative search capabilities',
          inputSchema: GENERATE_INPUT_SHAPE
        },
        async ({ query, collection, targetProperties, limit }: GenerateInput) => {
          const { name: targetCollection, schema } = await this.resolveCollection(collection);
          const validatedProperties = await this.validateTargetProperties(targetCollection, targetProperties, schema);
          const effectiveLimit = limit ?? 3;
          const rawResult = await this.weaviate.generateText(targetCollection, query, validatedProperties, effectiveLimit);
          const structured = this.safeParseJson(rawResult);
          const structuredContent = this.normalizeStructuredContent(structured);

          this.logger.info('Generate text success: result bytes=%d', rawResult.length);

          const response: {
            content: Array<{ type: 'text'; text: string }>;
            structuredContent?: Record<string, unknown>;
          } = {
            content: [
              {
                type: 'text',
                text: structuredContent ? JSON.stringify(structuredContent, null, 2) : rawResult
              }
            ]
          };

          if (structuredContent) {
            response.structuredContent = structuredContent;
          }

          return response;
        }
      );
    }
    
  }

  private registerResources(): void {
    const schemaTemplate = new ResourceTemplate(SCHEMA_URI_TEMPLATE, {
      list: async () => {
        try {
          const schema = await this.weaviate.getSchema();
          const classes = schema.classes ?? [];

          return {
            resources: classes.map((cls: any) => ({
              uri: `weaviate://schema/${cls.class}`,
              name: cls.class,
              description: cls.description ?? `Schema information for ${cls.class}`,
              mimeType: 'application/json'
            }))
          };
        } catch (error) {
          this.logger.error('Failed to list schema resources: %s', error instanceof Error ? error.message : String(error));
          return { resources: [] };
        }
      }
    });

    this.server.registerResource(
      SCHEMA_TEMPLATE_NAME,
      schemaTemplate,
      {
        title: 'Weaviate Collection Schema',
        description: 'Inspect schema for a specific Weaviate collection'
      },
      async (uri, variables) => {
        const rawCollection = Array.isArray(variables.collection)
          ? variables.collection[0]
          : variables.collection;
        let collectionName = rawCollection ?? '<unspecified>';

        try {
          const { name: resolvedCollection, schema } = await this.resolveCollection(rawCollection);
          collectionName = resolvedCollection;
          const body = JSON.stringify(schema, null, 2);

          return {
            contents: [
              {
                uri: uri.href,
                mimeType: 'application/json',
                text: body
              }
            ]
          };
        } catch (error) {
          const message = error instanceof Error ? error.message : String(error);
          this.logger.error('Failed to read schema %s: %s', collectionName, message);
          throw new Error(`Failed to read schema for collection ${collectionName}: ${message}`);
        }
      }
    );
  }

  private async validateTargetProperties(
    collection: string,
    targetProperties: string[],
    schema?: any
  ): Promise<string[]> {
    const uniqueProps = Array.from(new Set(targetProperties));
    const classSchema = schema ?? (await this.weaviate.getClassSchema(collection));
    const allowedProps = new Set((classSchema.properties ?? []).map((prop: any) => prop.name));

    for (const prop of uniqueProps) {
      if (!allowedProps.has(prop)) {
        throw new Error(`Property '${prop}' does not exist in collection '${collection}', what exists is: ${Array.from(allowedProps).join(', ')}`);
      }
    }

    return uniqueProps;
  }

  private async resolveCollection(candidate?: string | string[]): Promise<{ name: string; schema: any }> {
    const rawValue = Array.isArray(candidate) ? candidate[0] : candidate;
    const trimmed = rawValue?.trim();

    let collectionName = trimmed && trimmed.length > 0 ? trimmed : undefined;
    let availableCollections = await this.getCollectionNames();

    if (!collectionName) {
      if (availableCollections.length === 0) {
        throw new Error('No collections are available in the Weaviate schema.');
      }
      collectionName = availableCollections[0];
      this.logger.warn('No collection specified, defaulting to first available: %s', collectionName);
    }

    if (!availableCollections.includes(collectionName)) {
      availableCollections = await this.getCollectionNames(true);
    }

    if (!availableCollections.includes(collectionName)) {
      const namesList = availableCollections.length > 0 ? availableCollections.join(', ') : 'none';
      let descPart = '';
      try {
        const fullSchema = await this.weaviate.getSchema();
        const classes: Array<{ class: string; description?: string }> = Array.isArray(fullSchema.classes)
          ? fullSchema.classes
          : [];
        if (classes.length > 0) {
          const listWithDescriptions = classes
            .map((cls) => {
              const name = cls.class ?? '<unknown>';
              const desc = (cls.description ?? 'no description').toString();
              return `${name}: ${desc}`;
            })
            .join('; ');
          descPart = ` Descriptions: ${listWithDescriptions}`;
        }
      } catch {
        // ignore and use names only
      }
      throw new Error(
        `Collection '${collectionName}' was not found. Available collections: ${namesList}.${descPart}`
      );
    }

    const schema = await this.weaviate.getClassSchema(collectionName);
    return { name: collectionName, schema };
  }

  private async getCollectionNames(forceRefresh = false): Promise<string[]> {
    const now = Date.now();

    if (!forceRefresh && this.collectionCache && now - this.collectionCache.fetchedAt < COLLECTION_CACHE_TTL_MS) {
      return this.collectionCache.names;
    }

    const schema = await this.weaviate.getSchema();
    const names = Array.isArray(schema.classes) ? schema.classes.map((cls: any) => cls.class).filter(Boolean) : [];

    this.collectionCache = {
      names,
      fetchedAt: now
    };

    return names;
  }

  private safeParseJson(value: string): unknown | undefined {
    try {
      return JSON.parse(value);
    } catch {
      return undefined;
    }
  }

  private normalizeStructuredContent(value: unknown): Record<string, unknown> | undefined {
    if (value === null || value === undefined) {
      return undefined;
    }

    if (Array.isArray(value)) {
      return { results: value };
    }

    if (typeof value === 'object') {
      return value as Record<string, unknown>;
    }

    return undefined;
  }

  private isToolDisabled(toolName: string): boolean {
    return this.config.disabledTools.includes(toolName);
  }
}