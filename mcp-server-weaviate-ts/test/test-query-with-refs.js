#!/usr/bin/env node

import { ConfigLoader } from '../dist/config.js';
import { Logger } from '../dist/logger.js';
import { WeaviateConnection } from '../dist/weaviate.js';

async function main() {
  const config = ConfigLoader.loadConfig();
  const logger = new Logger(config);
  const conn = new WeaviateConnection(config, logger);

  const collection = process.env.TEST_COLLECTION || 'Etapa';
  const refProp = process.env.TEST_REF_PROP || 'belongsToFluxo'; // try also 'hasFicheiros'
  const q = process.env.TEST_QUERY || '5';
  const limit = Number(process.env.TEST_LIMIT || 5);

  const output = await conn.queryWithRefs(collection, refProp, q, limit, ['name'], ['name']);
  console.log('\nQuery with refs (dynamic) summary:');
  console.log(output);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
