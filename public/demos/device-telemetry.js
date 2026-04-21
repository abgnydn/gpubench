/**
 * Shared device telemetry — imported by all P2P demos.
 * Reports device info + workload stats to /api/device every 5s.
 *
 * Usage:
 *   <script src="device-telemetry.js"></script>
 *   // then in your update loop:
 *   reportDevice('Flappy Evolution', bestFitness, gen, speed);
 */

const _devId = localStorage.getItem('_devId') || (() => {
  const id = crypto.randomUUID();
  localStorage.setItem('_devId', id);
  return id;
})();

let _lastReport = 0;
let _cachedGpu = null;

function _detectGpu() {
  if (_cachedGpu) return Promise.resolve(_cachedGpu);
  return navigator.gpu?.requestAdapter?.()
    .then(a => a?.requestAdapterInfo?.() || a?.info)
    .then(i => {
      _cachedGpu = i?.description || i?.vendor || 'GPU';
      return _cachedGpu;
    })
    .catch(() => 'GPU');
}

function _deviceName() {
  const ua = navigator.userAgent;
  if (/iPhone|iPad|iPod|Android/i.test(ua)) return 'Mobile';
  if (ua.includes('Mac')) return 'Mac';
  return 'PC';
}

/**
 * @param {string} workload - e.g. "Flappy Evolution", "PETase (Island 0)"
 * @param {number} fitness  - current best fitness
 * @param {number} gen      - generation count
 * @param {number} speed    - gen/s
 */
function reportDevice(workload, fitness, gen, speed) {
  if (Date.now() - _lastReport < 5000) return;
  _lastReport = Date.now();

  const data = {
    id: _devId,
    name: _deviceName(),
    gpu: _cachedGpu || 'detecting',
    workload,
    fitness,
    gen,
    speed,
  };

  _detectGpu().then(gpu => { data.gpu = gpu; }).catch(() => {}).finally(() => {
    fetch('/api/device', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    })
      .then(r => {
        // Surface non-2xx responses (e.g. missing table, rate limit, bad body)
        // so failures are visible in devtools instead of silently swallowed.
        if (!r.ok) console.error('[device-telemetry] save failed:', r.status, r.statusText);
      })
      .catch(e => console.error('[device-telemetry] network error:', e));
  });
}
