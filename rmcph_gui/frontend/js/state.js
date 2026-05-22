// Minimal shared app state. As phases land, this holds the opened dataset, the
// parsed structure, the assembled k-path, and the active job id. A tiny
// pub/sub lets views react to changes without a framework.
'use strict';

const _state = {
    dataset: null,    // { eqFile, configsDir, nConfigs, atomDic, ... }  (Phase 1)
    structure: null,  // { lattice, atoms }                              (Phase 2)
    reciprocal: null, // { recipLattice, bzFacets, highSymPoints }       (Phase 3)
    kpath: null,      // { labels, coords, kstep }                       (Phase 3)
    job: null,        // { id, status, progress }                        (Phase 5)
};

const _subs = new Set();

export const state = {
    get: (key) => _state[key],
    set: (key, value) => {
        _state[key] = value;
        _subs.forEach((fn) => fn(key, value));
    },
    subscribe: (fn) => { _subs.add(fn); return () => _subs.delete(fn); },
};
