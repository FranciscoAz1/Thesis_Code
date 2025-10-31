import { Command } from 'commander';
import * as dotenv from 'dotenv';

// Load environment variables
dotenv.config();

export interface Config {
  // Weaviate connection
  weaviateHost: string;
  weaviateScheme: 'http' | 'https';

  // Server configuration
  transport: 'stdio' | 'http';
  httpPort: number;
  httpHost: string;

  // Logging
  logLevel: 'debug' | 'info' | 'warn' | 'error';
  logOutput: 'stderr' | 'file' | 'both';

  // Security
  readOnly: boolean;
  disabledTools: string[];

}

export class ConfigLoader {
  static loadConfig(): Config {
    const config: Config = {
      // Defaults
      weaviateHost: this.getEnvOrDefault('WEAVIATE_HOST', 'host.docker.internal:8080'),
      weaviateScheme: this.getEnvOrDefault('WEAVIATE_SCHEME', 'http') as 'http' | 'https',
      transport: this.getEnvOrDefault('MCP_TRANSPORT', 'stdio') as 'stdio' | 'http',
      httpPort: parseInt(this.getEnvOrDefault('MCP_HTTP_PORT', '3000')),
      httpHost: this.getEnvOrDefault('MCP_HTTP_HOST', '127.0.0.1'),
      logLevel: this.getEnvOrDefault('MCP_LOG_LEVEL', 'info') as 'debug' | 'info' | 'warn' | 'error',
      logOutput: this.getEnvOrDefault('MCP_LOG_OUTPUT', 'stderr') as 'stderr' | 'file' | 'both',
      readOnly: this.getEnvBool('MCP_READ_ONLY'),
      disabledTools: [],
    };

    // Parse disabled tools
    const disabled = process.env.MCP_DISABLED_TOOLS;
    if (disabled) {
      config.disabledTools = disabled.split(',').map(tool => tool.trim());
    }

    // Override with command-line arguments
    const program = new Command();
    program
      .option('--weaviate-host <host>', 'Weaviate host', config.weaviateHost)
      .option('--weaviate-scheme <scheme>', 'Weaviate scheme (http/https)', config.weaviateScheme)
      .option('--transport <transport>', 'Transport protocol (stdio/http)', config.transport)
      .option('--http-port <port>', 'HTTP port when using http transport', config.httpPort.toString())
      .option('--http-host <host>', 'HTTP host when using http transport', config.httpHost)
      .option('--log-level <level>', 'Log level (debug/info/warn/error)', config.logLevel)
      .option('--log-output <output>', 'Log output (stderr/file/both)', config.logOutput)
      .option('--read-only', 'Enable read-only mode')

    program.parse();
    const options = program.opts();

    // Apply command-line overrides
    Object.assign(config, {
      weaviateHost: options.weaviateHost || config.weaviateHost,
      weaviateScheme: options.weaviateScheme || config.weaviateScheme,
      transport: options.transport || config.transport,
      httpPort: options.httpPort ? parseInt(options.httpPort) : config.httpPort,
      httpHost: options.httpHost || config.httpHost,
      logLevel: options.logLevel || config.logLevel,
      logOutput: options.logOutput || config.logOutput,
      readOnly: options.readOnly !== undefined ? options.readOnly : config.readOnly,
    });

    // Validate configuration
    this.validateConfig(config);

    return config;
  }

  private static validateConfig(config: Config): void {
    if (!['stdio', 'http'].includes(config.transport)) {
      throw new Error(`Invalid transport: ${config.transport}, must be 'stdio' or 'http'`);
    }

    if (!['debug', 'info', 'warn', 'error'].includes(config.logLevel)) {
      throw new Error(`Invalid log level: ${config.logLevel}`);
    }

    if (!['stderr', 'file', 'both'].includes(config.logOutput)) {
      throw new Error(`Invalid log output: ${config.logOutput}`);
    }

    if (!['http', 'https'].includes(config.weaviateScheme)) {
      throw new Error(`Invalid Weaviate scheme: ${config.weaviateScheme}`);
    }
  }

  static isToolDisabled(config: Config, toolName: string): boolean {
    return config.disabledTools.includes(toolName);
  }

  private static getEnvOrDefault(key: string, defaultValue: string): string {
    return process.env[key] || defaultValue;
  }

  private static getEnvBool(key: string): boolean {
    const value = process.env[key];
    return value === 'true' || value === '1' || value === 'yes';
  }
}