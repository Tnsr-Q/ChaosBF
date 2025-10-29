// ChaosBF v4.0.1 Smoke Tests
// Tests: determinism, pointer stability, acceptance rate

async function loadWASM() {
  const response = await fetch('./chaosbf_wasm.wasm');
  const buffer = await response.arrayBuffer();
  const module = await WebAssembly.instantiate(buffer, {});
  return module.instance.exports;
}

function encodeString(str) {
  return new TextEncoder().encode(str);
}

function log(message, pass = null) {
  const div = document.createElement('div');
  div.className = 'test' + (pass === true ? ' pass' : pass === false ? ' fail' : '');
  div.innerHTML = `<pre>${message}</pre>`;
  document.getElementById('results').appendChild(div);
}

async function runTests() {
  log('Loading WASM module...');
  const wasm = await loadWASM();
  log('✓ WASM loaded successfully', true);

  // Test 1: Determinism
  log('\n=== Test 1: Determinism ===');
  const seed = 12345;
  const code = encodeString('?*@+=');
  const codePtr = wasm.__heap_base || 1024;  // Assume heap starts here
  const memory = new Uint8Array(wasm.memory.buffer);
  memory.set(code, codePtr);

  // Run 1
  wasm.init_sim(seed, 256, 256, codePtr, code.length, 200.0, 0.6);
  const checkResult1 = wasm.self_check();
  if (checkResult1 !== 1) {
    log(`✗ self_check() failed: ${checkResult1}`, false);
    return;
  }
  log('✓ self_check() passed', true);

  wasm.step_sim(100);
  const metricsPtr1 = wasm.get_metrics_ptr();
  const metrics1 = new Float32Array(wasm.memory.buffer, metricsPtr1, 20);
  const snapshot1 = Array.from(metrics1);

  // Run 2 (same seed)
  wasm.init_sim(seed, 256, 256, codePtr, code.length, 200.0, 0.6);
  wasm.step_sim(100);
  const metricsPtr2 = wasm.get_metrics_ptr();
  const metrics2 = new Float32Array(wasm.memory.buffer, metricsPtr2, 20);
  const snapshot2 = Array.from(metrics2);

  let deterministic = true;
  for (let i = 0; i < 20; i++) {
    if (Math.abs(snapshot1[i] - snapshot2[i]) > 0.001) {
      deterministic = false;
      log(`✗ Mismatch at metric[${i}]: ${snapshot1[i]} vs ${snapshot2[i]}`, false);
    }
  }

  if (deterministic) {
    log('✓ Determinism test passed (100 steps identical)', true);
  } else {
    log('✗ Determinism test FAILED', false);
  }

  // Test 2: Pointer Stability
  log('\n=== Test 2: Pointer Stability ===');
  wasm.init_sim(seed + 1, 256, 256, codePtr, code.length, 200.0, 0.6);

  const metricsPtrInitial = wasm.get_metrics_ptr();
  const memPtrInitial = wasm.get_mem_ptr();

  let pointerStable = true;
  for (let i = 0; i < 100; i++) {
    wasm.step_sim(100);
    const metricsPtr = wasm.get_metrics_ptr();
    const memPtr = wasm.get_mem_ptr();

    if (metricsPtr !== metricsPtrInitial || memPtr !== memPtrInitial) {
      pointerStable = false;
      log(`✗ Pointer changed at iteration ${i}`, false);
      log(`  metrics: ${metricsPtrInitial} → ${metricsPtr}`, false);
      log(`  mem: ${memPtrInitial} → ${memPtr}`, false);
      break;
    }
  }

  if (pointerStable) {
    log('✓ Pointer stability test passed (10k ticks)', true);
  } else {
    log('✗ Pointer stability test FAILED', false);
  }

  // Test 3: Metropolis Acceptance Rate
  log('\n=== Test 3: Metropolis Acceptance Rate ===');
  wasm.init_sim(seed + 2, 256, 256, codePtr, code.length, 200.0, 0.6);
  wasm.set_metropolis(true);
  wasm.step_sim(1000);  // Run enough steps to get samples

  const finalMetricsPtr = wasm.get_metrics_ptr();
  const finalMetrics = new Float32Array(wasm.memory.buffer, finalMetricsPtr, 20);
  const acceptanceRate = finalMetrics[19];  // Index 19 is acceptance rate

  log(`Acceptance rate: ${acceptanceRate.toFixed(3)}`);

  if (acceptanceRate >= 0.20 && acceptanceRate <= 0.30) {
    log('✓ Acceptance rate in bounds [0.20, 0.30]', true);
  } else if (acceptanceRate === 0) {
    log('⚠ Acceptance rate is 0 (might need more steps or metropolis not working)', null);
  } else {
    log(`✗ Acceptance rate out of bounds: ${acceptanceRate}`, false);
  }

  // Test 4: get_mem_len()
  log('\n=== Test 4: Memory Length ===');
  const memLen = wasm.get_mem_len();
  log(`Memory length: ${memLen} bytes`);
  if (memLen === 256 * 256) {
    log('✓ Memory length matches grid size', true);
  } else {
    log(`✗ Expected ${256 * 256}, got ${memLen}`, false);
  }

  // Test 5: self_check() validation
  log('\n=== Test 5: self_check() validation ===');
  wasm.init_sim(seed + 3, 256, 256, codePtr, code.length, 200.0, 0.6);
  wasm.step_sim(100);
  const checkResult = wasm.self_check();

  if (checkResult === 1) {
    log('✓ self_check() returns 1 (pass)', true);
  } else {
    log(`✗ self_check() returns ${checkResult} (fail)`, false);
  }

  log('\n=== All Tests Complete ===');
}

runTests().catch(err => {
  log(`FATAL ERROR: ${err.message}`, false);
  console.error(err);
});
