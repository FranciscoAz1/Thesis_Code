import * as fs from 'fs';
import * as path from 'path';
import { Config } from './config.js';

export enum LogLevel {
  DEBUG = 0,
  INFO = 1,
  WARN = 2,
  ERROR = 3,
}

export class Logger {
  private level: LogLevel;
  private outputs: Array<(message: string) => void> = [];

  constructor(config: Config) {
    this.level = this.parseLogLevel(config.logLevel);
    this.setupOutputs(config.logOutput);
  }

  private parseLogLevel(level: string): LogLevel {
    switch (level.toLowerCase()) {
      case 'debug':
        return LogLevel.DEBUG;
      case 'info':
        return LogLevel.INFO;
      case 'warn':
      case 'warning':
        return LogLevel.WARN;
      case 'error':
        return LogLevel.ERROR;
      default:
        return LogLevel.INFO;
    }
  }

  private setupOutputs(logOutput: string): void {
    switch (logOutput) {
      case 'stderr':
        this.outputs.push((message: string) => {
          process.stderr.write(message);
        });
        break;
      case 'file':
        this.setupFileOutput();
        break;
      case 'both':
        this.outputs.push((message: string) => {
          process.stderr.write(message);
        });
        this.setupFileOutput();
        break;
      default:
        this.outputs.push((message: string) => {
          process.stderr.write(message);
        });
    }
  }

  private setupFileOutput(): void {
    const logDir = 'logs';
    
    // Create logs directory if it doesn't exist
    if (!fs.existsSync(logDir)) {
      try {
        fs.mkdirSync(logDir, { recursive: true });
      } catch (err) {
        console.error(`Failed to create log directory: ${err}`);
        return;
      }
    }

    const logFilePath = path.join(logDir, 'mcp-server.log');
    
    try {
      const logFile = fs.createWriteStream(logFilePath, { flags: 'a' });
      this.outputs.push((message: string) => {
        logFile.write(message);
      });
    } catch (err) {
      console.error(`Failed to open log file: ${err}`);
    }
  }

  debug(format: string, ...args: any[]): void {
    this.log(LogLevel.DEBUG, format, ...args);
  }

  info(format: string, ...args: any[]): void {
    this.log(LogLevel.INFO, format, ...args);
  }

  warn(format: string, ...args: any[]): void {
    this.log(LogLevel.WARN, format, ...args);
  }

  error(format: string, ...args: any[]): void {
    this.log(LogLevel.ERROR, format, ...args);
  }

  private log(level: LogLevel, format: string, ...args: any[]): void {
    if (level < this.level) {
      return;
    }

    const message = this.formatMessage(format, ...args);
    const levelStr = this.levelToString(level);
    const timestamp = new Date().toISOString();
    const logLine = `[${timestamp}] ${levelStr}: MCP-Server: ${message}\n`;

    this.outputs.forEach(output => {
      try {
        output(logLine);
      } catch (err) {
        // Fallback to console if output fails
        console.error(`Logger error: ${err}`);
        console.log(logLine);
      }
    });
  }

  private formatMessage(format: string, ...args: any[]): string {
    if (args.length === 0) {
      return format;
    }

    // Simple sprintf-like formatting
    return format.replace(/%[sdv%]/g, (match) => {
      if (match === '%%') return '%';
      if (args.length === 0) return match;
      
      const arg = args.shift();
      switch (match) {
        case '%s':
          return String(arg);
        case '%d':
          return String(Number(arg));
        case '%v':
          return String(arg);
        default:
          return match;
      }
    });
  }

  private levelToString(level: LogLevel): string {
    switch (level) {
      case LogLevel.DEBUG:
        return 'DEBUG';
      case LogLevel.INFO:
        return 'INFO';
      case LogLevel.WARN:
        return 'WARN';
      case LogLevel.ERROR:
        return 'ERROR';
      default:
        return 'UNKNOWN';
    }
  }
}