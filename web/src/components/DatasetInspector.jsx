import React from 'react';
import { File, AlertTriangle, CheckCircle, Database } from 'lucide-react';

export default function DatasetInspector({ directoryName, filesList, configFamily, baseStructure }) {
  if (!directoryName) return null;

  return (
    <div className="mt-4 bg-black/40 rounded-xl border border-white/5 overflow-hidden">
      <div className="px-4 py-3 border-b border-white/10 bg-white/5 flex items-center justify-between">
        <h3 className="text-sm font-medium text-gray-300 flex items-center gap-2">
          <Database className="w-4 h-4 text-indigo-400" />
          {directoryName}
        </h3>
        <span className="text-xs bg-blue-500/20 text-blue-400 px-2 py-1 rounded-full border border-blue-500/20">
          {configFamily === 'rmc6f' ? '.rmc6f Ensemble' : 'Frac Ensemble'}
        </span>
      </div>

      <div className="p-4 space-y-4">
        {/* Metadata section */}
        {baseStructure && (
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="bg-white/5 p-2 rounded border border-white/5">
              <span className="text-gray-500 block mb-1">Basis sites (→ {3 * Object.values(baseStructure.atomDic).reduce((a, b) => a + b.length, 0)} bands)</span>
              <span className="font-mono text-gray-200">
                {Object.values(baseStructure.atomDic).reduce((a, b) => a + b.length, 0)}
              </span>
            </div>
            <div className="bg-white/5 p-2 rounded border border-white/5">
              <span className="text-gray-500 block mb-1">Supercell Dim</span>
              <span className="font-mono text-gray-200">
                {baseStructure.dim ? baseStructure.dim.join('x') : 'N/A'}
              </span>
            </div>
            <div className="bg-white/5 p-2 rounded border border-white/5 col-span-2">
              <span className="text-gray-500 block mb-1">Elements</span>
              <div className="flex gap-1 flex-wrap">
                {Object.entries(baseStructure.atomDic).map(([el, arr]) => (
                  <span key={el} className="bg-indigo-500/20 text-indigo-300 px-1.5 py-0.5 rounded border border-indigo-500/30">
                    {el}: {arr.length}
                  </span>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* File List Section */}
        <div className="max-h-48 overflow-y-auto pr-2 custom-scrollbar space-y-1">
          {filesList.length === 0 ? (
            <div className="flex items-center gap-2 text-amber-500 text-xs bg-amber-500/10 p-2 rounded border border-amber-500/20">
              <AlertTriangle className="w-4 h-4" />
              <span>No compatible files found in this directory.</span>
            </div>
          ) : (
            <div className="flex flex-col gap-1">
              <div className="text-xs text-gray-500 mb-1 flex items-center gap-2">
                <CheckCircle className="w-4 h-4 text-emerald-500" />
                Found {filesList.length} configurations
              </div>
              {filesList.slice(0, 50).map((f, i) => (
                <div key={i} className="flex items-center gap-2 text-xs text-gray-400 bg-white/5 p-1.5 rounded">
                  <File className="w-3 h-3 text-gray-500" />
                  <span className="truncate">{f.name}</span>
                </div>
              ))}
              {filesList.length > 50 && (
                <div className="text-xs text-center text-gray-500 pt-2">
                  + {filesList.length - 50} more files
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
