#!/usr/bin/env node

import { spawn } from 'child_process';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { LATEST_PROTOCOL_VERSION } from '@modelcontextprotocol/sdk/types.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

async function testMCPServer() {
  console.log('Testing MCP Server...');

  // Start the server
  const serverProcess = spawn('node', [join(__dirname, '..', 'dist', 'main.js')], {
    stdio: ['pipe', 'pipe', 'pipe'],
    env: {
      ...process.env,
      // Force stdio so the test can talk JSON-RPC over stdin/stdout regardless of the user's env
      MCP_TRANSPORT: 'stdio',
      // Optional: increase verbosity if needed
      // MCP_LOG_LEVEL: 'debug',
    },
  });

  let nextRequestId = 1;

  const handshakeMessages = [
    {
      jsonrpc: '2.0',
      id: 0,
      method: 'initialize',
      params: {
        protocolVersion: LATEST_PROTOCOL_VERSION,
        capabilities: {
          tools: { listChanged: true },
          resources: { listChanged: true }
        },
        clientInfo: {
          name: 'weaviate-mcp-test',
          version: '0.0.1'
        }
      }
    },
    {
      jsonrpc: '2.0',
      method: 'initialized',
      params: {}
    }
  ];

  const testRequests = [
    {
      jsonrpc: '2.0',
      id: nextRequestId++,
      method: 'tools/list',
      params: {}
    },
    {
      jsonrpc: '2.0',
      id: nextRequestId++,
      method: 'resources/list',
      params: {}
    },
    {
      jsonrpc: '2.0',
      id: nextRequestId++,
      method: 'tools/call',
      params: {
        name: 'weaviate-query',
        arguments: {
          query: 'hello',
          collection: 'Etapa',
          targetProperties: ['name'],
          limit: 1
        }
      }
    },
    // New: origin-style summary starting from Etapa
    {
      jsonrpc: '2.0',
      id: nextRequestId++,
      method: 'tools/call',
      params: {
        name: 'weaviate-origin',
        arguments: {
          query: 'What is the News',
          collection: 'Etapa',
          targetProperties: ['name'],
          limit: 1
        }
      }
    },
    // New: follow a specific reference from Etapa -> belongsToFluxo
    // {
    //   jsonrpc: '2.0',
    //   id: nextRequestId++,
    //   method: 'tools/call',
    //   params: {
    //     name: 'weaviate-follow-ref',
    //     arguments: {
    //       query: '5',
    //       collection: 'Etapa',
    //       refProp: 'belongsToFlux',
    //       baseProps: ['name'],
    //       refProps: ['name'],
    //       limit: 1
    //     }
    //   }
    // },
    // // Negative test: invalid collection to ensure we get a proper error response and the list of available collections
    // {
    //   jsonrpc: '2.0',
    //   id: nextRequestId++,
    //   method: 'tools/call',
    //   params: {
    //     name: 'weaviate-query',
    //     arguments: {
    //       query: 'show me something',
    //       collection: 'NonExistentCollection',
    //       targetProperties: ['text'],
    //       limit: 1
    //     }
    //   }
    // },
    // {
    //   jsonrpc: '2.0',
    //   id: nextRequestId++,
    //   method: 'tools/call',
    //   params: {
    //     name: 'weaviate-generate-text',
    //     arguments: {
    //       query: 'hello',
    //       collection: 'Dataset',
    //       targetProperties: ['text', 'file_path'],
    //       limit: 1
    //     }
    //   }
    // },
    // {
    //   jsonrpc: '2.0',
    //   id: nextRequestId++,
    //   method: 'tools/call',
    //   params: {
    //     name: 'weaviate-generate-text',
    //     arguments: {
    //       query: 'hello',
    //       collection: 'Dataset',
    //       targetProperties: ['text', 'file_path'],
    //       limit: 1
    //     }
    //   }
    // }
  ];


  // Send handshake then test requests one-by-one.
  const expectedResponseIds = new Set(testRequests.filter((msg) => typeof msg.id === 'number').map((msg) => msg.id));
  expectedResponseIds.add(0); // initialize response

  let currentTestIndex = 0;
  // waitingForId: id of the request we expect a response for before sending the next.
  let waitingForId = null;

  function sendInitialize() {
    const msg = handshakeMessages[0];
    console.log(`\nSending initialize: ${JSON.stringify(msg)}`);
    serverProcess.stdin.write(JSON.stringify(msg) + '\n');
    waitingForId = msg.id; // expect response with id 0
  }

  function sendInitializedNotification() {
    const msg = handshakeMessages[1];
    console.log(`\nSending initialized notification`);
    serverProcess.stdin.write(JSON.stringify(msg) + '\n');
    // initialized is a notification (no id) so send the first test request immediately
    sendNextTestRequest();
  }

  function sendNextTestRequest() {
    if (currentTestIndex >= testRequests.length) {
      // nothing left to send; waitingForId should be null or the last id
      return;
    }
    const msg = testRequests[currentTestIndex++];
    console.log(`\nSending test request: ${JSON.stringify(msg)}`);
    serverProcess.stdin.write(JSON.stringify(msg) + '\n');
    // expect a response with this id before sending the next
    waitingForId = msg.id;
  }

  // Start the sequence by sending initialize
  sendInitialize();

  // Listen for responses and trigger the next send when appropriate
  serverProcess.stdout.on('data', (data) => {
    const responses = data.toString().trim().split('\n');
    for (const response of responses) {
      if (!response) continue;
      console.log(`Response: ${response}`);
      try {
        const parsed = JSON.parse(response);
        if (parsed.id !== undefined) {
          // Mark this id as received
          expectedResponseIds.delete(parsed.id);

          // If this was the initialize response (id 0), send initialized notification
          if (parsed.id === 0) {
            // clear waitingForId for initialize response
            if (waitingForId === 0) waitingForId = null;
            sendInitializedNotification();
            continue; // proceed to next response if any
          }

          // If this response matches the id we're waiting for, send next test request
          if (waitingForId !== null && parsed.id === waitingForId) {
            waitingForId = null;
            // If there are more test requests, send the next one
            sendNextTestRequest();
          }
        }
      } catch (error) {
        console.error('Failed to parse response JSON', error);
      }

      if (expectedResponseIds.size === 0) {
        console.log('\nTest completed successfully!');
        serverProcess.kill();
        process.exit(0);
      }
    }
  });

  serverProcess.stderr.on('data', (data) => {
    console.log(`Server log: ${data.toString()}`);
  });

  serverProcess.on('close', (code) => {
    console.log(`Server process exited with code ${code}`);
  });

  // Timeout after 100 seconds
  setTimeout(() => {
    console.log('Test timeout - killing server');
    serverProcess.kill();
    process.exit(1);
  }, 100000);
}

testMCPServer().catch(console.error);