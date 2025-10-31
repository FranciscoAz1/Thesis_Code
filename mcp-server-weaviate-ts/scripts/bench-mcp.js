#!/usr/bin/env node

// Simple MCP benchmark over HTTP.
// Usage (PowerShell):
//   $env:MCP_URL="http://127.0.0.1:3000/mcp" ; node scripts/bench-mcp.js
// Optional env:
//   MCP_URL: HTTP endpoint (default http://127.0.0.1:3000/mcp)
//   MCP_TOOL: weaviate-query | weaviate-generate-text (default weaviate-query)
//   MCP_COLLECTION: collection/class name (default from server, or 'Dataset')
//   MCP_PROPS: comma-separated properties (default "text,file_path")
//   MCP_QUERY: query string (default "hello")
//   MCP_LIMIT: integer limit (default 1)
//   MCP_ITERS: number of iterations (default 5)
//
// Prints latency stats and minimal error summary.

const MCP_URL = process.env.MCP_URL || 'http://127.0.0.1:3000/mcp';
const TOOL = process.env.MCP_TOOL || 'weaviate-query';
const COLLECTION = process.env.MCP_COLLECTION || 'LiHuaWorld';
const PROPS = (process.env.MCP_PROPS || 'text,file_path').split(',').map(s => s.trim()).filter(Boolean);
const QUERY = process.env.MCP_QUERY || 'hello';
const LIMIT = Number(process.env.MCP_LIMIT || 1);
const ITERS = Number(process.env.MCP_ITERS || 5);

function now() { return performance.now(); }

async function rpc(method, params, id = 1) {
  const body = { jsonrpc: '2.0', id, method, params };
  const res = await fetch(MCP_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const json = await res.json();
  if (json.error) throw new Error(`RPC error ${json.error.code}: ${json.error.message}`);
  return json.result;
}

async function initializeOnce() {
  return rpc('initialize', {
    protocolVersion: '2025-06-18',
    capabilities: { tools: { listChanged: true }, resources: { listChanged: true } },
    clientInfo: { name: 'bench-client', version: '0.0.1' }
  }, 1);
}

async function toolsCallOnce(toolName, args, idBase) {
  return rpc('tools/call', { name: toolName, arguments: args }, idBase);
}

function summarize(times) {
  const sorted = [...times].sort((a,b)=>a-b);
  const n = sorted.length;
  const sum = sorted.reduce((acc,x)=>acc+x,0);
  const avg = sum / n;
  const p50 = sorted[Math.floor(0.50*(n-1))];
  const p90 = sorted[Math.floor(0.90*(n-1))];
  const p95 = sorted[Math.floor(0.95*(n-1))];
  const p99 = sorted[Math.floor(0.99*(n-1))];
  return { n, min: sorted[0], p50, avg, p90, p95, p99, max: sorted[n-1] };
}

(async () => {
  console.log(`Benchmarking MCP at ${MCP_URL}`);
  console.log(`Tool=${TOOL} collection=${COLLECTION} props=${PROPS.join(', ')} query="${QUERY}" limit=${LIMIT} iters=${ITERS}`);

  // Initialize session
  try {
    const t0 = now();
    const info = await initializeOnce();
    const t1 = now();
    console.log(`Initialized in ${(t1 - t0).toFixed(1)} ms; server=${info.serverInfo?.name} v${info.serverInfo?.version}`);
  } catch (e) {
    console.error('Failed to initialize:', e.message);
    process.exit(1);
  }

  const args = { query: QUERY, collection: COLLECTION, targetProperties: PROPS, limit: LIMIT };
  const times = [];
  let errors = 0;

  for (let i = 0; i < ITERS; i++) {
    const start = now();
    try {
      const result = await toolsCallOnce(TOOL, args, 100 + i);
      const end = now();
      times.push(end - start);
      if (result.isError) {
        errors++;
        console.warn(`Iter ${i+1}: tool returned error content`);
      }
      if (i === 0) {
        const preview = JSON.stringify(result).slice(0, 240).replace(/\n/g,' ');
        console.log(`Sample result: ${preview}...`);
      }
    } catch (e) {
      const end = now();
      times.push(end - start);
      errors++;
      console.warn(`Iter ${i+1} failed: ${e.message}`);
    }
  }

  const s = summarize(times);
  console.log('\nLatency (ms):');
  console.log(`  n=${s.n}, min=${s.min.toFixed(1)}, p50=${s.p50.toFixed(1)}, avg=${s.avg.toFixed(1)}, p90=${s.p90.toFixed(1)}, p95=${s.p95.toFixed(1)}, p99=${s.p99.toFixed(1)}, max=${s.max.toFixed(1)}`);
  console.log(`Errors: ${errors}/${ITERS}`);
})();
