#!/usr/bin/env node

import { createRequire } from 'node:module';
import { ConfigLoader } from './config.js';
import { Logger } from './logger.js';
import { WeaviateMcpRuntime } from './mcp.js';

const require = createRequire(import.meta.url);
const pkg = require('../package.json');
const VERSION = typeof pkg?.version === 'string' ? pkg.version : '0.0.0';

async function main(): Promise<void> {
  try {
    // Load configuration
    const config = ConfigLoader.loadConfig();
    
    // Initialize logger
    const logger = new Logger(config);
    logger.info('Starting Weaviate MCP Server v0.1.0');
    logger.info(
      `Configuration: host=${config.weaviateHost}, scheme=${config.weaviateScheme}, transport=${config.transport}, read-only=${config.readOnly}`
    );

    // Create MCP runtime
    const runtime = new WeaviateMcpRuntime(config, logger, VERSION);

    // Handle graceful shutdown
    const handleShutdown = async (signal: string) => {
      logger.info(`Received ${signal}, shutting down server...`);
      try {
        await runtime.shutdown();
      } catch (shutdownError) {
        logger.error('Error during shutdown: %s', shutdownError instanceof Error ? shutdownError.message : String(shutdownError));
      } finally {
        process.exit(0);
      }
    };

    process.on('SIGINT', () => {
      void handleShutdown('SIGINT');
    });
    process.on('SIGTERM', () => {
      void handleShutdown('SIGTERM');
    });

    // Start server based on transport
    switch (config.transport) {
      case 'stdio':
        logger.info('Starting server with stdio transport');
        await runtime.startStdio();
        break;
      case 'http':
        logger.info(`Starting server with HTTP transport on ${config.httpHost}:${config.httpPort}`);
        await runtime.startHttp(config.httpHost, config.httpPort);
        break;
      default:
        logger.error(`Unsupported transport: ${config.transport}`);
        process.exit(1);
    }
  } catch (error) {
    console.error(`Failed to start server: ${error}`);
    process.exit(1);
  }
}

// Handle unhandled promise rejections
process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
  process.exit(1);
});

// Handle uncaught exceptions
process.on('uncaughtException', (error) => {
  console.error('Uncaught Exception:', error);
  process.exit(1);
});

main().catch((error) => {
  console.error(`Application error: ${error}`);
  process.exit(1);
});