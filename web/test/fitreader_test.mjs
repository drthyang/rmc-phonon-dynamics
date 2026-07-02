// web/test/fitreader_test.mjs
//
// Classification + Rw guards for the comprehensive fit-quality reader
// (src/io/sqgr.js). Uses a mock File System Access directory handle so the
// pure grouping/dedup logic is testable in Node.
//
// Pins the RMCProfile filename → channel mapping (ported from rmc-toolkits):
//   _XFQ / _FQ → X-ray S(Q) (deduped, _XFQ wins), _FT_XFQ → X-ray G(r),
//   _SQ → neutron S(Q), _PDF<n> → neutron G(r), _bragg → Bragg,
//   *partials → ignored (no fit channel).

import { listSqgrConfigs, getConfigChannels, configRw, channelTag } from '../src/io/sqgr.js';

let fails = 0;
const ok = (c, m) => { console.log(`  ${c ? 'ok  ' : 'FAIL'} ${m}`); if (!c) fails++; };

// Mock a directory handle whose files carry CSV text. calc=col2, exp=col3.
const CSV_RW100 = 'Q,calc,exp\n1,0,1\n2,0,1\n';   // Rw = sqrt(2/2)*100 = 100
const CSV_RW0 = 'Q,calc,exp\n1,1,1\n2,1,1\n';     // Rw = 0
const mkFile = (name, text = CSV_RW100) => ({ kind: 'file', name, getFile: async () => ({ text: async () => text }) });
const mkDir = (files) => ({ async *values() { for (const f of files) yield f; } });

// ── classification + dedup ──────────────────────────────────────────────────
const dir = mkDir([
  // config 1 — X-ray, with the _XFQ/_FQ twin and both partials kinds
  mkFile('GTS_5K_1_XFQ1.csv', CSV_RW100),
  mkFile('GTS_5K_1_FQ1.csv', CSV_RW0),          // same X-ray data, lower precision → deduped out
  mkFile('GTS_5K_1_FT_XFQ1.csv'),
  mkFile('GTS_5K_1_PDFpartials.csv'),           // partials → ignored
  mkFile('GTS_5K_1_FQ1partials.csv'),           // partials → ignored
  // config 2 — neutron + Bragg + a second neutron PDF bank
  mkFile('GTS_5K_2_SQ1.csv'),
  mkFile('GTS_5K_2_PDF1.csv'),
  mkFile('GTS_5K_2_PDF2.csv'),
  mkFile('GTS_5K_2_bragg.csv'),
]);

const ents = await listSqgrConfigs(dir);
ok(ents.length === 2, `two configs grouped (got ${ents.length})`);

const c1 = ents.find(e => e.config === 1), c2 = ents.find(e => e.config === 2);
const slots1 = c1 ? Object.keys(c1.channels).sort() : [];
ok(slots1.join(',') === 'xgr,xsq', `config 1 → X-ray S(Q)+G(r) only, partials excluded (got ${slots1.join(',')})`);
ok(c1?.channels.xsq.handle.name === 'GTS_5K_1_XFQ1.csv', `_XFQ wins over _FQ dedup (got ${c1?.channels.xsq.handle.name})`);

const slots2 = c2 ? Object.keys(c2.channels).sort() : [];
ok(slots2.join(',') === 'bragg,ngr,ngr#2,nsq', `config 2 → neutron S(Q), G(r) banks 1&2, Bragg (got ${slots2.join(',')})`);
ok(c2?.channels['ngr#2'].bank === 2, 'second neutron PDF bank captured as ngr#2');

// ── Rw dedup: config 1 S(Q) must read the _XFQ file (Rw 100), not _FQ (Rw 0) ──
const rw1 = await configRw(c1);
const xsqPart = rw1.parts.find(p => p.slot === 'xsq');
ok(xsqPart && Math.abs(xsqPart.rw - 100) < 1e-9, `X-ray S(Q) Rw from _XFQ file = 100 (got ${xsqPart?.rw})`);
ok(rw1.parts.length === 2, `config 1 has two Rw channels, no partials (got ${rw1.parts.length})`);

// ── channel curves + tags ────────────────────────────────────────────────────
const ch2 = await getConfigChannels(c2);
const tags = ch2.map(c => c.tag);
ok(tags.join(' | ') === 'N S(Q) | N G(r) | N G(r)·2 | Bragg', `channel tags + order (got ${tags.join(' | ')})`);
ok(ch2.every(c => c.x.length === 2 && c.rmc.length === 2 && c.expt.length === 2), 'curves parsed (x/rmc/expt)');
ok(channelTag({ group: 'X-ray', quantity: 'G(r)' }) === 'X G(r)', 'channelTag X-ray G(r)');

// ── Bragg ToF axis detection ─────────────────────────────────────────────────
const tofDir = mkDir([mkFile('S_1_bragg.csv', 'Flight time (us),calc,exp\n1000,0,1\n2000,0,1\n')]);
const tofEnt = (await listSqgrConfigs(tofDir))[0];
const tofCh = (await getConfigChannels(tofEnt))[0];
ok(tofCh.xlabel === 'ToF (µs)', `Bragg ToF header → ToF axis (got ${tofCh.xlabel})`);

console.log(`\n${fails === 0 ? '✅ fit reader OK' : `❌ ${fails} failed`}`);
process.exit(fails ? 1 : 0);
