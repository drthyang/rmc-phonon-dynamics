// web/src/compute/engine.js
import wgslCode from './Sk_kernel.wgsl?raw';

export class ComputeEngine {
  constructor() {
    this.device = null;
    this.pipeline = null;
  }

  async init() {
    if (!navigator.gpu) {
      throw new Error("WebGPU not supported on this browser.");
    }
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      throw new Error("No appropriate GPU adapter found.");
    }
    this.device = await adapter.requestDevice();

    const shaderModule = this.device.createShaderModule({
      label: 'Sk_kernel',
      code: wgslCode,
    });

    this.pipeline = this.device.createComputePipeline({
      label: 'Sk_pipeline',
      layout: 'auto',
      compute: {
        module: shaderModule,
        entryPoint: 'main',
      },
    });
  }

  /**
   * Computes the S(k) matrices for a batch of frames.
   * @param {Array} kvec - [kx, ky, kz]
   * @param {Float32Array} displacements - [N * Frames * 3]
   * @param {Float32Array} cellIdx - [N * Frames * 3]
   * @param {Float32Array} masses - [N]
   * @param {Uint32Array} typeIndices - [N]
   * @param {number} numTypes - T
   * @param {number} numFrames - F
   * @param {Float32Array} counts - [T] count of atoms per type
   * @returns {Object} { Sk_real: Float64Array(3T x 3T), Sk_imag: Float64Array(3T x 3T) }
   */
  async computeBatch(kvec, displacements, cellIdx, masses, typeIndices, numTypes, numFrames, counts) {
    if (!this.device) await this.init();

    const numAtoms = masses.length;
    const totalThreads = numAtoms * numFrames;
    
    // Uniform buffer. WGSL struct { kvec: vec3<f32>, numAtoms: u32, numTypes: u32 }
    // is 20 bytes; allocate 32 (16-byte aligned) for safety.
    const uniformBufferAligned = this.device.createBuffer({
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const alignedUniforms = new ArrayBuffer(32);
    const alignedF32 = new Float32Array(alignedUniforms);
    const alignedU32 = new Uint32Array(alignedUniforms);
    alignedF32[0] = kvec[0];
    alignedF32[1] = kvec[1];
    alignedF32[2] = kvec[2];
    alignedU32[3] = numAtoms;
    alignedU32[4] = numTypes;
    this.device.queue.writeBuffer(uniformBufferAligned, 0, alignedUniforms);

    const dispBuffer = this.device.createBuffer({
      size: displacements.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(dispBuffer, 0, displacements);

    const cellBuffer = this.device.createBuffer({
      size: cellIdx.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(cellBuffer, 0, cellIdx);

    const massBuffer = this.device.createBuffer({
      size: masses.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(massBuffer, 0, masses);

    const typeBuffer = this.device.createBuffer({
      size: typeIndices.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(typeBuffer, 0, typeIndices);

    // outAB is [Frames, numTypes * 6] elements. Size in bytes:
    const outElements = numFrames * numTypes * 6;
    const outBufferSize = outElements * 4;
    const outBuffer = this.device.createBuffer({
      size: outBufferSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    
    // Clear the outBuffer to 0 (atomic additions need 0-initialized)
    const zeroData = new Uint32Array(outElements);
    this.device.queue.writeBuffer(outBuffer, 0, zeroData);

    // Create bind group
    const bindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBufferAligned } },
        { binding: 1, resource: { buffer: dispBuffer } },
        { binding: 2, resource: { buffer: cellBuffer } },
        { binding: 3, resource: { buffer: massBuffer } },
        { binding: 4, resource: { buffer: typeBuffer } },
        { binding: 5, resource: { buffer: outBuffer } },
      ],
    });

    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(this.pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(totalThreads / 256));
    passEncoder.end();

    // Read back buffer
    const readBuffer = this.device.createBuffer({
      size: outBufferSize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    commandEncoder.copyBufferToBuffer(outBuffer, 0, readBuffer, 0, outBufferSize);

    this.device.queue.submit([commandEncoder.finish()]);

    await readBuffer.mapAsync(GPUMapMode.READ);
    const outData = new Float32Array(readBuffer.getMappedRange());
    
    const dimSize = numTypes * 3;
    const Sk_real = new Float64Array(dimSize * dimSize);
    const Sk_imag = new Float64Array(dimSize * dimSize);

    // Process output and accumulate outer products
    const norm = new Float32Array(numTypes);
    for (let t = 0; t < numTypes; t++) {
      norm[t] = 1.0 / Math.sqrt(Math.max(counts[t], 1.0));
    }

    for (let f = 0; f < numFrames; f++) {
      const A = new Float32Array(dimSize);
      const B = new Float32Array(dimSize);
      
      for (let t = 0; t < numTypes; t++) {
        const base = f * numTypes * 6 + t * 6;
        const n = norm[t];
        A[t * 3 + 0] = outData[base + 0] * n;
        A[t * 3 + 1] = outData[base + 1] * n;
        A[t * 3 + 2] = outData[base + 2] * n;
        B[t * 3 + 0] = outData[base + 3] * n;
        B[t * 3 + 1] = outData[base + 4] * n;
        B[t * 3 + 2] = outData[base + 5] * n;
      }
      
      for (let i = 0; i < dimSize; i++) {
        for (let j = 0; j < dimSize; j++) {
          const idx = i * dimSize + j;
          Sk_real[idx] += (A[i] * A[j] + B[i] * B[j]);
          Sk_imag[idx] += (B[i] * A[j] - A[i] * B[j]);
        }
      }
    }

    readBuffer.unmap();
    
    // Cleanup WebGPU resources
    uniformBufferAligned.destroy();
    dispBuffer.destroy();
    cellBuffer.destroy();
    massBuffer.destroy();
    typeBuffer.destroy();
    outBuffer.destroy();
    readBuffer.destroy();

    return { Sk_real, Sk_imag };
  }
}
