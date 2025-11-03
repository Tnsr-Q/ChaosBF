// ChaosBF Web Worker - runs simulation in background thread
let wasm = null;
let wasmMemory = null;
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
      await initSimulation(payload);
      break;
  }
};

async function initSimulation(config) {
  try {
    // Load WASM module
    if (!wasm) {
      const response = await fetch('./chaosbf_wasm.wasm');
      const bytes = await response.arrayBuffer();
      const result = await WebAssembly.instantiate(bytes, {});
      wasm = result.instance.exports;
      wasmMemory = wasm.memory;
    }

    width = config.width;
    height = config.height;

    // Encode code to bytes
    const encoder = new TextEncoder();
    const codeBytes = encoder.encode(config.code || '?*@+=');

    // Use __heap_base.value when available, fallback to 1024
    const heapBase = wasm.__heap_base ? Number(wasm.__heap_base.value) : 1024;
    const codePtr = heapBase;
    
    // Validate capacity before writing to memory
    const memory = new Uint8Array(wasmMemory.buffer);
    if (codePtr + codeBytes.length > memory.length) {
      throw new Error(`Insufficient memory: need ${codePtr + codeBytes.length} bytes, have ${memory.length}`);
    }
    memory.set(codeBytes, codePtr);

    // Initialize simulation with BigInt seed for i64 parameter
    wasm.init_sim(
      config.seed || BigInt(Date.now()),
      width,
      height,
      codePtr,
      codeBytes.length,
      config.e0 || 200.0,
      config.t0 || 0.6
    );

    // Run self-check
    const checkResult = wasm.self_check();
    if (checkResult !== 1) {
      self.postMessage({
        type: 'error',
        error: `self_check() failed: ${checkResult}`
      });
      return;
    }

    self.postMessage({ type: 'ready' });
  } catch (err) {
    self.postMessage({ type: 'error', error: err.message });
  }
}

function simulationLoop() {
  if (!running || !wasm) {
    return;
  }

  try {
    // Step simulation
    wasm.step_sim(ticksPerFrame);

    // Read metrics
    const metricsPtr = wasm.get_metrics_ptr();
    const metricsArray = new Float32Array(wasmMemory.buffer, metricsPtr, 20);

    const metrics = {
      step: metricsArray[0],
      e: metricsArray[1],
      t: metricsArray[2],
      s: metricsArray[3],
      f: metricsArray[4],
      lambda_hat: metricsArray[5],
      mutations: metricsArray[6],
      replications: metricsArray[7],
      crossovers: metricsArray[8],
      learns: metricsArray[9],
      lambda_volatility: metricsArray[10],
      ds_dt_ema: metricsArray[11],
      dk_dt_ema: metricsArray[12],
      complexity_estimate: metricsArray[13],
      info_per_energy: metricsArray[14],
      genome_bank_size: metricsArray[15],
      output_len: metricsArray[16],
      pid_kp: metricsArray[17],
      variance_gamma: metricsArray[18],
      acceptance_rate: metricsArray[19],
    };

    // Read tape memory
    const memPtr = wasm.get_mem_ptr();
    const memLen = wasm.get_mem_len();
    const tape = new Uint8Array(wasmMemory.buffer, memPtr, memLen);

    // Copy frame data (don't transfer ownership - it's WASM memory)
    const frame = new Uint8Array(tape);

    self.postMessage({ type: 'frame', frame, metrics });

    // Continue loop
    if (running) {
      setTimeout(() => simulationLoop(), 16);  // ~60 FPS
    }
  } catch (err) {
    self.postMessage({ type: 'error', error: err.message });
    running = false;
  }
}
