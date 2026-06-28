import { analyzeBravais, primToConv, fracToCart } from '../src/math/bravais.js';
import { buildBZModel } from '../src/math/highsym.js';

let fail = 0;
function test(name, A_conv, basis, expectCode) {
  const br = analyzeBravais(A_conv, basis);
  const m = buildBZModel(br);
  const facesBySides = m.bz.faces.reduce((a, f) => { a[f.length] = (a[f.length] || 0) + 1; return a; }, {});
  const codeOk = br.code === expectCode;
  console.log(`${codeOk ? 'ok  ' : 'FAIL'} ${name}: ${br.code} (${br.system}/${br.centering}) BZ ${m.bz.vertices.length}v ${m.bz.faces.length}f ${JSON.stringify(facesBySides)}`);
  if (!codeOk) fail++;
  // Cartesian k-vector preserved under prim->conv transform:
  //   frac_prim via B_prim  ==  frac_conv via B_conv  (Å⁻¹).
  let maxErr = 0;
  for (const [, p] of Object.entries(m.points)) {
    const c1 = p.cart;                                  // = fracToCart(frac, B_prim)
    const c2 = fracToCart(primToConv(p.frac, br.T), br.B_conv);
    maxErr = Math.max(maxErr, Math.hypot(c1[0] - c2[0], c1[1] - c2[1], c1[2] - c2[2]));
  }
  const ok = maxErr < 1e-9;
  console.log(`     ${ok ? 'ok  ' : 'FAIL'} cartesian k-vector preserved (maxErr=${maxErr.toExponential(2)})`);
  if (!ok) fail++;
}
const a = 4.0, D = [[a, 0, 0], [0, a, 0], [0, 0, a]];
test('Simple cubic', D, [{ el: 'A', frac: [0, 0, 0] }], 'CUB');
test('FCC', D, [{ el: 'A', frac: [0, 0, 0] }, { el: 'A', frac: [0, .5, .5] }, { el: 'A', frac: [.5, 0, .5] }, { el: 'A', frac: [.5, .5, 0] }], 'FCC');
test('BCC', D, [{ el: 'A', frac: [0, 0, 0] }, { el: 'A', frac: [.5, .5, .5] }], 'BCC');
test('Hexagonal', [[3, 0, 0], [-1.5, 2.598, 0], [0, 0, 5]], [{ el: 'A', frac: [0, 0, 0] }], 'HEX');
console.log(fail === 0 ? '\n✅ BZ checks passed' : `\n❌ ${fail} failed`);
process.exit(fail ? 1 : 0);
