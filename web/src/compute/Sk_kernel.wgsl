// web/src/compute/Sk_kernel.wgsl

struct Uniforms {
    kvec: vec3<f32>,
    numAtoms: u32,
    numTypes: u32,
}

@group(0) @binding(0) var<uniform> params: Uniforms;
@group(0) @binding(1) var<storage, read> displacements: array<f32>; // packed [N, 3] per frame, flattened across frames
@group(0) @binding(2) var<storage, read> cellIdx: array<f32>;       // packed [N, 3]
@group(0) @binding(3) var<storage, read> masses: array<f32>;        // [N]
@group(0) @binding(4) var<storage, read> typeIndices: array<u32>;   // [N]
@group(0) @binding(5) var<storage, read_write> outAB: array<atomic<u32>>; // [Frames, numTypes * 6] (3 real, 3 imag)

fn atomicAddFloat(dest: ptr<storage, atomic<u32>, read_write>, val: f32) {
    var old_u32 = atomicLoad(dest);
    loop {
        let old_f32 = bitcast<f32>(old_u32);
        let new_f32 = old_f32 + val;
        let new_u32 = bitcast<u32>(new_f32);
        let res = atomicCompareExchangeWeak(dest, old_u32, new_u32);
        if (res.exchanged) {
            break;
        }
        old_u32 = res.old_value;
    }
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x; // index into flat displacements array (1 idx = 1 atom in 1 frame)
    let N = params.numAtoms;
    
    // total size of displacements array in atoms is (displacements.length / 3).
    // Let's assume idx is the flat atom index across all frames in the batch.
    let frameIdx = idx / N;
    let atomIdx = idx % N;
    
    // Safety check: ensure we don't read out of bounds if array length isn't perfectly divisible by 256
    let totalAtoms = arrayLength(&masses) * (arrayLength(&displacements) / (arrayLength(&masses) * 3));
    if (idx >= u32(arrayLength(&displacements) / 3)) {
        return;
    }

    let tIndex = typeIndices[atomIdx];
    let mass = masses[atomIdx];
    let sqrt_m = sqrt(mass);

    let baseDisp = idx * 3;
    let dispX = displacements[baseDisp + 0];
    let dispY = displacements[baseDisp + 1];
    let dispZ = displacements[baseDisp + 2];

    let baseCell = idx * 3;
    let cellX = cellIdx[baseCell + 0];
    let cellY = cellIdx[baseCell + 1];
    let cellZ = cellIdx[baseCell + 2];

    let kx = params.kvec.x;
    let ky = params.kvec.y;
    let kz = params.kvec.z;

    let dot_prod = cellX * kx + cellY * ky + cellZ * kz;
    let cos_p = cos(dot_prod);
    let sin_p = sin(dot_prod);

    let wd_real_x = dispX * sqrt_m * cos_p;
    let wd_real_y = dispY * sqrt_m * cos_p;
    let wd_real_z = dispZ * sqrt_m * cos_p;

    let wd_imag_x = dispX * sqrt_m * sin_p;
    let wd_imag_y = dispY * sqrt_m * sin_p;
    let wd_imag_z = dispZ * sqrt_m * sin_p;

    let outBase = frameIdx * params.numTypes * 6 + tIndex * 6;
    
    // Real parts (A)
    atomicAddFloat(&outAB[outBase + 0], wd_real_x);
    atomicAddFloat(&outAB[outBase + 1], wd_real_y);
    atomicAddFloat(&outAB[outBase + 2], wd_real_z);
    
    // Imag parts (B)
    atomicAddFloat(&outAB[outBase + 3], wd_imag_x);
    atomicAddFloat(&outAB[outBase + 4], wd_imag_y);
    atomicAddFloat(&outAB[outBase + 5], wd_imag_z);
}
