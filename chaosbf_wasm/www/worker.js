// ChaosBF Web Worker - runs simulation in background thread
let wasm = null;
let frame = null;
let frameBuffer = null;
let width = 0;
let height = 0;
let running = false;
let ticksPerFrame = 100; // adjustable performance tuning

self.onmessage = async (e) => {
  const { type, payload } = e.data;

  switch (type) {
    case 'init':
      await initSimulation(payload);
      break;

    case 'start':
      running = true;
      requestAnimationFrame(simulationLoop);
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
  // Load WASM module
  if (!wasm) {
    const wasmModule = await import('../pkg/chaosbf_wasm.js');
    await wasmModule.default();
    wasm = wasmModule;
  }

  width = config.width;
  height = config.height;

  // Allocate frame buffer (transferable)
  frameBuffer = new Uint8Array(width * height);
  frame = new Uint8Array(frameBuffer.buffer);

  // Initialize simulation
  wasm.init_sim({
    seed: config.seed || Date.now(),
    width,
    height,
    code: config.code,
    e0: config.e0 || 200.0,
    t0: config.t0 || 0.6,
    theta_rep: config.theta_rep || 6.0,
    landauer_win: config.landauer_win || 16,
  });

  self.postMessage({ type: 'ready' });
}

function simulationLoop() {
  if (!running || !wasm) {
    return;
  }

  try {
    // Step simulation and write frame into our buffer
    const metrics = wasm.step_sim(ticksPerFrame, frame);

    // Send frame to main thread (transfer ownership for zero-copy)
    self.postMessage(
      {
        type: 'frame',
        frame: frameBuffer,
        metrics,
      },
      [frameBuffer.buffer]
    );

    // Recreate buffer for next frame (previous was transferred)
    frameBuffer = new Uint8Array(width * height);
    frame = new Uint8Array(frameBuffer.buffer);

    // Continue loop
    requestAnimationFrame(simulationLoop);
  } catch (err) {
    self.postMessage({ type: 'error', error: err.message });
    running = false;
  }
}

// Fallback for environments without requestAnimationFrame
if (typeof requestAnimationFrame === 'undefined') {
  self.requestAnimationFrame = (cb) => setTimeout(cb, 16);
}
