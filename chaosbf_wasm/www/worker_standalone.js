// ChaosBF Web Worker - Standalone WASM version
let wasm = null;
let memory = null;
let width = 0;
let height = 0;
let running = false;
let ticksPerFrame = 100;

self.onmessage = async (e) => {
  const { type, payload } = e.data;

  switch (type) {
    case 'init':
      await initSimulation(payload);
      break;

    case 'start':
      running = true;
      simulationLoop();
      break;

    case 'stop':
      running = false;
      break;

    case 'setSpeed':
      ticksPerFrame = payload.ticksPerFrame;
      break;

    case 'reset':
      await initSimulation(payload.config);
      break;
  }
};

async function initSimulation(config) {
  width = config.width;
  height = config.height;

  // Load WASM module
  const response = await fetch('./pkg/chaosbf_wasm.wasm');
  const bytes = await response.arrayBuffer();
  const wasmModule = await WebAssembly.instantiate(bytes);

  wasm = wasmModule.instance.exports;
  memory = wasm.memory;

  // Convert code string to bytes
  const encoder = new TextEncoder();
  const codeBytes = encoder.encode(config.code);

  // Allocate space in WASM memory for code
  const codePtr = new Uint8Array(memory.buffer, 0, codeBytes.length);
  codePtr.set(codeBytes);

  // Initialize simulation
  wasm.init_sim(
    BigInt(config.seed || Date.now()),
    width,
    height,
    0, // code pointer (at start of memory)
    codeBytes.length,
    config.e0 || 200.0,
    config.t0 || 0.6
  );

  self.postMessage({ type: 'ready' });
}

function simulationLoop() {
  if (!running || !wasm) {
    return;
  }

  try {
    // Step simulation
    wasm.step_sim(ticksPerFrame);

    // Get frame from WASM memory
    const memPtr = wasm.get_mem_ptr();
    const frameSize = width * height;
    const frame = new Uint8Array(memory.buffer, memPtr, frameSize);

    // Get metrics
    const metricsPtr = wasm.get_metrics_ptr();
    const metricsF32 = new Float32Array(memory.buffer, metricsPtr, 10);

    const metrics = {
      step: Math.round(metricsF32[0]),
      e: metricsF32[1],
      t: metricsF32[2],
      s: metricsF32[3],
      f: metricsF32[4],
      lambda_hat: metricsF32[5],
      mutations: Math.round(metricsF32[6]),
      replications: Math.round(metricsF32[7]),
      crossovers: Math.round(metricsF32[8]),
      learns: Math.round(metricsF32[9]),
      genome_bank_size: 0, // not exposed in standalone
      elite_count: 0,
    };

    // Copy frame data
    const frameCopy = new Uint8Array(frame);

    // Send to main thread
    self.postMessage({
      type: 'frame',
      frame: frameCopy,
      metrics,
    });

    // Continue loop
    setTimeout(() => simulationLoop(), 0);
  } catch (err) {
    self.postMessage({ type: 'error', error: err.message });
    running = false;
  }
}
