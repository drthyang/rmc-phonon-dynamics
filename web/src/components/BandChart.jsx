import React, { useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

/**
 * Native React Band Structure Chart using Recharts
 * @param {Array} bands - The connected bands array: [kpoints][modes]
 */
export default function BandChart({ bands, onPointClick, kpathMeta }) {
  const data = useMemo(() => {
    if (!bands || bands.length === 0) return [];

    const numModes = bands[0].length;
    const hsymIndex = kpathMeta?.hsymIndex || {};
    const chartData = [];

    for (let k = 0; k < bands.length; k++) {
      // Label only flat-q indices that are high-symmetry points (segment ends).
      const name = hsymIndex[k] !== undefined ? hsymIndex[k] : '';
      const point = { name, kIndex: k };
      for (let m = 0; m < numModes; m++) {
        point[`mode_${m}`] = bands[k][m];
      }
      chartData.push(point);
    }
    return chartData;
  }, [bands, kpathMeta]);

  if (!bands || bands.length === 0) return null;
  const numModes = bands[0].length;

  return (
    <div className="w-full h-full relative" style={{ minHeight: '400px' }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
          <XAxis dataKey="name" stroke="#888" tick={{ fill: '#888' }} />
          <YAxis stroke="#888" tick={{ fill: '#888' }} label={{ value: 'Energy (meV)', angle: -90, position: 'insideLeft', fill: '#888' }} />
          <Tooltip 
            contentStyle={{ backgroundColor: 'rgba(24, 24, 27, 0.9)', borderColor: 'rgba(255,255,255,0.1)', color: '#fff' }}
            itemStyle={{ color: '#fff' }}
          />
          {Array.from({ length: numModes }).map((_, m) => (
            <Line 
              key={`mode_${m}`}
              type="monotone" 
              dataKey={`mode_${m}`} 
              stroke={`hsl(${(m * 137.5) % 360}, 70%, 60%)`} 
              dot={false}
              activeDot={{ r: 6, onClick: (_, payload) => {
                if (onPointClick) {
                  onPointClick(payload.payload.kIndex, m);
                }
              }}}
              isAnimationActive={false}
              strokeWidth={2}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
      <div className="absolute top-2 left-4 text-xs text-gray-400 bg-black/50 px-2 py-1 rounded">
        Click on any point to visualize the 3D atomic mode
      </div>
    </div>
  );
}
