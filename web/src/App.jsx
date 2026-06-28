import { useState, useRef, useEffect } from 'react';
import { Activity, Cpu, Cog, Eye } from 'lucide-react';
import { PhononPipeline } from './compute/pipeline';
import { fromResults } from './io/viewermodel';
import { generatePhonopyBandYaml, downloadString } from './io/writers';
import RunnerPage from './pages/RunnerPage';
import ViewerPage from './pages/ViewerPage';

export default function App() {
  const [page, setPage] = useState('runner');     // 'runner' | 'viewer'
  const [model, setModel] = useState(null);        // unified viewer model
  const [ready, setReady] = useState(false);
  const pipelineRef = useRef(null);

  useEffect(() => {
    const p = new PhononPipeline(() => {});
    pipelineRef.current = p;
    p.initWorkers(4).then(() => setReady(true)).catch(e => { console.error('Init failed:', e); setReady(true); });
  }, []);

  const onResults = (results, kpathMeta) => {
    const m = fromResults(results, kpathMeta);
    setModel(m);
    setPage('viewer');
    // Auto-save the band.yaml after a successful run.
    try {
      const yaml = generatePhonopyBandYaml(m.baseStructure, m.qPoints, m.bands, m.eigvecs, m.kpathMeta);
      downloadString(yaml, 'band_gpu.yaml', 'text/yaml');
    } catch (e) { console.error('band.yaml auto-save failed:', e); }
  };

  const loadModel = (m) => { setModel(m); setPage('viewer'); };

  return (
    <div className="min-h-screen bg-black text-gray-100 flex flex-col font-sans selection:bg-blue-500/30">
      <nav className="h-16 border-b border-white/10 flex items-center px-6 justify-between glass-panel sticky top-0 z-50">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center shadow-lg shadow-blue-500/20">
            <Activity className="w-5 h-5 text-white" />
          </div>
          <h1 className="text-xl font-semibold tracking-tight">RMC Phonon Dynamics</h1>
          <div className="ml-6 flex rounded-lg overflow-hidden border border-white/10">
            <button onClick={() => setPage('runner')} className={`flex items-center gap-1.5 px-4 py-1.5 text-sm ${page === 'runner' ? 'bg-blue-600 text-white' : 'bg-white/5 text-gray-300 hover:bg-white/10'}`}>
              <Cog className="w-4 h-4" />Runner
            </button>
            <button onClick={() => setPage('viewer')} className={`flex items-center gap-1.5 px-4 py-1.5 text-sm ${page === 'viewer' ? 'bg-blue-600 text-white' : 'bg-white/5 text-gray-300 hover:bg-white/10'}`}>
              <Eye className="w-4 h-4" />Viewer
            </button>
          </div>
        </div>
        <div className="flex items-center gap-2 bg-blue-500/10 text-blue-400 px-3 py-1 rounded-full border border-blue-500/20 text-sm font-medium">
          <Cpu className="w-4 h-4" />
          <span>{ready ? 'WebGPU Ready' : 'Initializing…'}</span>
        </div>
      </nav>

      <main className="flex-1 max-w-[1400px] w-full mx-auto p-6">
        {page === 'runner' && <RunnerPage pipeline={pipelineRef.current} onResults={onResults} onLoadResult={loadModel} />}
        {page === 'viewer' && <ViewerPage model={model} onLoadModel={setModel} />}
      </main>
    </div>
  );
}
