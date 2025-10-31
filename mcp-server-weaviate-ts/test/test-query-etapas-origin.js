#!/usr/bin/env node

import { ConfigLoader } from '../dist/config.js';
import { Logger } from '../dist/logger.js';
import { WeaviateConnection } from '../dist/weaviate.js';

async function main() {
  const config = ConfigLoader.loadConfig();
  const logger = new Logger(config);
  const conn = new WeaviateConnection(config, logger);

  const q = process.env.TEST_QUERY || '5';
  const limit = Number(process.env.TEST_LIMIT || 5);
  const collection = 'Etapa';
  const nl = await conn.queryOrigin(collection, q, limit, ['name']);
  console.log(nl);
}
main().catch((err) => {
  console.error(err);
  process.exit(1);
});
