// web/src/io/symmetryWorker.js
//
// Dedicated off-main-thread Web Worker for streaming RMC supercell ensemble
// symmetry, domain correlation, and zone-boundary subgroup discovery.
//
// Message Protocol (strictly backpressured 1 frame in flight at a time):
//   main -> worker: { type: 'init', family, baseStructure, options }
//   worker -> main: { type: 'ready' }
//   main -> worker: { type: 'frame', text, name }
//   worker -> main: { type: 'ack', accepted, nConfigs }
//   main -> worker: { type: 'finalize', totalAvailableConfigs }
//   worker -> main: { type: 'result', result }

import { createStreamingSymmetrySession } from '../compute/supercellAnalyzer.js';

let session = null;

if (typeof self !== 'undefined' && typeof self.postMessage === 'function') {
  self.onmessage = (ev) => {
    const msg = ev.data;
    try {
      if (msg.type === 'init') {
        session = createStreamingSymmetrySession(msg.family, msg.baseStructure, msg.options);
        self.postMessage({ type: 'ready' });
        return;
      }
      if (msg.type === 'frame') {
        if (!session) throw new Error('Symmetry worker received frame before init.');
        const info = session.pushText(msg.text, msg.name);
        self.postMessage({ type: 'ack', ...info });
        return;
      }
      if (msg.type === 'finalize') {
        if (!session) throw new Error('Symmetry worker received finalize before init.');
        const result = session.finalize(msg.totalAvailableConfigs || 1);
        session = null;
        self.postMessage({ type: 'result', result });
      }
    } catch (err) {
      self.postMessage({ type: 'error', error: err?.message || String(err) });
    }
  };
}
