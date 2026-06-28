import { useState, useRef, useEffect } from 'react'
import { FolderOpen, Play, Settings, Database, Activity, Cpu } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import { listConfigs, readBaseStructure, findStructureFile } from './io/readers'
import { PhononPipeline } from './compute/pipeline'
import { detectSystem, highSymmetryPoints, buildKPath } from './math/reciprocal'
import BandChart from './components/BandChart'
import CrystalViewer from './components/CrystalViewer'
import DatasetInspector from './components/DatasetInspector'
import BrillouinZoneViewer from './components/BrillouinZoneViewer'
import InsPanel from './components/InsPanel'
import { generatePhonopyBandYaml, downloadString } from './io/writers'

export default function App() {
  const [directoryHandle, setDirectoryHandle] = useState(null)
  const [filesList, setFilesList] = useState([])
  const [configFamily, setConfigFamily] = useState(null)
  const [baseStructure, setBaseStructure] = useState(null)
  
  const [isProcessing, setIsProcessing] = useState(false)
  const [progress, setProgress] = useState(0)
  const [progressText, setProgressText] = useState("")
  
  const [temperature, setTemperature] = useState(5)
  const [kstep, setKstep] = useState(20)
  const [results, setResults] = useState(null)
  const [kpathMeta, setKpathMeta] = useState(null)

  // Interaction State
  const [selectedKIndex, setSelectedKIndex] = useState(0)
  const [selectedModeIndex, setSelectedModeIndex] = useState(0)
  const [selectedPath, setSelectedPath] = useState([])
  const [bzPoints, setBzPoints] = useState({})

  // Crystal system + high-symmetry points derived from the loaded structure.
  const crystalInfo = baseStructure?.v1
    ? detectSystem(baseStructure.v1, baseStructure.v2, baseStructure.v3, baseStructure.dim)
    : null
  const symSet = crystalInfo ? highSymmetryPoints(crystalInfo.system) : null

  const pipelineRef = useRef(null)

  useEffect(() => {
    pipelineRef.current = new PhononPipeline((prog, text) => {
      setProgress(prog)
      setProgressText(text)
    })
    pipelineRef.current.initWorkers(4).catch(e => console.error("Worker init failed:", e))
  }, [])

  const handleSelectFolder = async () => {
    try {
      const dirHandle = await window.showDirectoryPicker({ mode: 'read' })
      setDirectoryHandle(dirHandle)
      
      const { files, family } = await listConfigs(dirHandle)
      setFilesList(files)
      setConfigFamily(family)

      if (family === 'rmc6f' && files.length > 0) {
        const baseInfo = await readBaseStructure(files[0])
        setBaseStructure(baseInfo)
      } else if (family === 'frac') {
        // Frac*.txt configs have no lattice/RN->element info; read it from a
        // companion .rmc6f structure file in the same directory.
        const structHandle = await findStructureFile(dirHandle)
        if (!structHandle) {
          setProgressText("Frac configs found, but no .rmc6f structure file in this folder to read the lattice/atoms from.")
          setBaseStructure(null)
          return
        }
        const baseInfo = await readBaseStructure(structHandle)
        setBaseStructure(baseInfo)
      } else {
        setBaseStructure(null)
      }
    } catch (err) {
      console.error(err)
    }
  }

  const handleRunCalculation = async () => {
    if (!filesList.length || !baseStructure) return
    setIsProcessing(true)
    setProgress(0)
    setProgressText("Starting...")
    
    try {
      if (selectedPath.length < 2) {
         setProgressText("Please select at least 2 points in the K-Path.");
         setIsProcessing(false);
         return;
      }

      // Build the k-path (fractional q + per-segment sizes + label index).
      // The pipeline applies the 2*pi Bloch-phase factor internally.
      const { qFrac, segSizes, hsymIndex } = buildKPath(bzPoints, selectedPath, kstep)
      setKpathMeta({ qFrac, segSizes, hsymIndex, pathLabels: selectedPath, kstep })

      const res = await pipelineRef.current.runCalculation(
        filesList, configFamily, baseStructure, qFrac, temperature, 50
      )

      setResults(res)

      // baseStructure is enriched by the pipeline (hsym_xyz, atomType, uniqueRN, ...)
      setBaseStructure(res.baseStructure)

      setProgressText("Calculation Complete!")
      setTimeout(() => setIsProcessing(false), 2000)
    } catch (e) {
      console.error(e)
      setProgressText("Error: " + e.message)
      setIsProcessing(false)
    }
  }

  const activeEigenvector = results?.eigvecs?.[selectedKIndex]?.[selectedModeIndex] || null;

  return (
    <div className="min-h-screen bg-black text-gray-100 flex flex-col font-sans selection:bg-blue-500/30">
      
      <nav className="h-16 border-b border-white/10 flex items-center px-6 justify-between glass-panel sticky top-0 z-50">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center shadow-lg shadow-blue-500/20">
            <Activity className="w-5 h-5 text-white" />
          </div>
          <h1 className="text-xl font-semibold tracking-tight">RMC Phonon Dynamics</h1>
        </div>
        <div className="flex items-center gap-4 text-sm font-medium text-gray-400">
          <div className="flex items-center gap-2 bg-blue-500/10 text-blue-400 px-3 py-1 rounded-full border border-blue-500/20">
            <Cpu className="w-4 h-4" />
            <span>WebGPU Ready</span>
          </div>
        </div>
      </nav>

      <main className="flex-1 max-w-7xl w-full mx-auto p-6 grid grid-cols-12 gap-6">
        
        {/* Left Sidebar (Controls) */}
        <div className="col-span-12 lg:col-span-3 flex flex-col gap-6">
          
          <motion.div 
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="glass-panel rounded-2xl p-6"
          >
            <div className="flex items-center gap-3 mb-6 text-gray-200">
              <Database className="w-5 h-5 text-indigo-400" />
              <h2 className="text-lg font-medium">Dataset</h2>
            </div>
            
            <button 
              onClick={handleSelectFolder}
              className="w-full flex items-center justify-center gap-2 bg-white/5 hover:bg-white/10 border border-white/10 transition-colors py-3 px-4 rounded-xl font-medium"
            >
              <FolderOpen className="w-5 h-5" />
              {directoryHandle ? 'Change Directory' : 'Select Directory'}
            </button>
            
            <DatasetInspector 
              directoryName={directoryHandle?.name}
              filesList={filesList}
              configFamily={configFamily}
              baseStructure={baseStructure}
            />
          </motion.div>

          <motion.div 
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.15 }}
            className={`glass-panel rounded-2xl h-[300px] transition-all ${baseStructure ? 'opacity-100' : 'opacity-50 pointer-events-none'}`}
          >
            <BrillouinZoneViewer
               symSet={symSet}
               system={crystalInfo?.system}
               onPathChange={(path, points) => {
                  setSelectedPath(path);
                  setBzPoints(points);
               }}
            />
          </motion.div>

          <motion.div 
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className={`glass-panel rounded-2xl p-6 transition-all ${baseStructure ? 'opacity-100' : 'opacity-50 pointer-events-none'}`}
          >
            <div className="flex items-center gap-3 mb-6 text-gray-200">
              <Settings className="w-5 h-5 text-indigo-400" />
              <h2 className="text-lg font-medium">Settings</h2>
            </div>

            <div className="space-y-4 mb-8">
              {crystalInfo && (
                <div className="text-xs text-gray-400">
                  Detected system: <span className="text-indigo-300 font-mono">{crystalInfo.system}</span>
                  {' '}(a,b,c = {crystalInfo.a.toFixed(2)}, {crystalInfo.b.toFixed(2)}, {crystalInfo.c.toFixed(2)} Å)
                </div>
              )}
              <div>
                <label className="text-xs text-gray-400 uppercase tracking-wider mb-2 block">Temperature (K)</label>
                <input
                  type="number"
                  value={temperature}
                  onChange={e => setTemperature(parseFloat(e.target.value))}
                  className="w-full bg-white/5 border border-white/10 rounded-lg px-4 py-2 focus:outline-none focus:border-blue-500 transition-colors"
                />
              </div>
              <div>
                <label className="text-xs text-gray-400 uppercase tracking-wider mb-2 block">Points per segment</label>
                <input
                  type="number"
                  value={kstep}
                  min={2}
                  onChange={e => setKstep(Math.max(2, parseInt(e.target.value) || 2))}
                  className="w-full bg-white/5 border border-white/10 rounded-lg px-4 py-2 focus:outline-none focus:border-blue-500 transition-colors"
                />
              </div>
            </div>

            <button 
              onClick={handleRunCalculation}
              disabled={isProcessing}
              className={`w-full flex items-center justify-center gap-2 py-3 px-4 rounded-xl font-medium transition-all ${
                isProcessing 
                  ? 'bg-blue-600/50 cursor-not-allowed' 
                  : 'bg-blue-600 hover:bg-blue-500 shadow-lg shadow-blue-500/25'
              }`}
            >
              {isProcessing ? (
                <>
                  <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  Processing...
                </>
              ) : (
                <>
                  <Play className="w-5 h-5" />
                  Run Calculation
                </>
              )}
            </button>
            
            <AnimatePresence>
              {(isProcessing || progressText === "Calculation Complete!") && (
                <motion.div 
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  className="mt-6"
                >
                  <div className="flex justify-between text-xs text-gray-400 mb-2">
                    <span className="truncate mr-4">{progressText}</span>
                    <span>{Math.round(progress)}%</span>
                  </div>
                  <div className="w-full bg-white/10 h-2 rounded-full overflow-hidden">
                    <motion.div 
                      className="h-full bg-blue-500"
                      initial={{ width: 0 }}
                      animate={{ width: `${progress}%` }}
                      transition={{ duration: 0.2 }}
                    />
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>

        </div>

        {/* Right Content (Native Viewer & Charts) */}
        <div className="col-span-12 lg:col-span-9 flex flex-col gap-6">
          
          <motion.div 
            initial={{ opacity: 0, scale: 0.98 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.2 }}
            className="glass-panel rounded-2xl h-[400px] flex items-center justify-center relative overflow-hidden"
          >
            {results ? (
              <>
              <button
                onClick={() => {
                  const yaml = generatePhonopyBandYaml(results.baseStructure, results.qPoints, results.bands, results.eigvecs, kpathMeta)
                  downloadString(yaml, 'band_gpu.yaml')
                }}
                className="absolute top-3 right-3 z-20 text-xs bg-white/10 hover:bg-white/20 px-3 py-1 rounded border border-white/10"
              >
                Export band.yaml
              </button>
              <BandChart
                bands={results.bands}
                kpathMeta={kpathMeta}
                onPointClick={(kIndex, modeIndex) => {
                  setSelectedKIndex(kIndex);
                  setSelectedModeIndex(modeIndex);
                }}
              />
              </>
            ) : (
              <>
                <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,rgba(59,130,246,0.1),transparent_50%)] pointer-events-none" />
                <div className="text-center z-10">
                  <h3 className="text-xl font-medium text-gray-300 mb-2">Band Structure</h3>
                  <p className="text-gray-500 text-sm max-w-sm mx-auto">
                    Run the calculation to visualize the continuous connected phonon bands natively.
                  </p>
                </div>
              </>
            )}
          </motion.div>

          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="glass-panel rounded-2xl h-[400px] relative overflow-hidden"
          >
            {results && baseStructure ? (
              <>
                <CrystalViewer
                  baseStructure={baseStructure}
                  eigenvector={activeEigenvector}
                  qPoint={results.qPoints?.[selectedKIndex]}
                  isPlaying={true}
                  amplitude={3.0}
                />
                <div className="absolute bottom-4 left-4 glass-panel px-4 py-2 rounded-lg text-sm font-mono text-gray-300 pointer-events-none">
                  Mode: {selectedModeIndex + 1} | Energy: {results.bands[selectedKIndex][selectedModeIndex].toFixed(2)} meV | k-point: {selectedKIndex + 1}
                </div>
              </>
            ) : (
               <div className="absolute inset-0 flex items-center justify-center">
                  <p className="text-gray-500 text-sm">3D Mode Visualization (Waiting for data...)</p>
               </div>
            )}
          </motion.div>

          {results && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="glass-panel rounded-2xl p-6"
            >
              <InsPanel results={results} kpathMeta={kpathMeta} temperature={temperature} />
            </motion.div>
          )}

        </div>

      </main>
    </div>
  )
}
