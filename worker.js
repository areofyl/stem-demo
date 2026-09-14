// Web Worker — runs one stem separation in its own thread
importScripts('stem.js');

let module = null;

self.onmessage = async function(e) {
  const { type, modelIdx, modelData, audioData, nSamples } = e.data;

  if (type === 'init') {
    module = await StemModule();
    // load model weights
    const ptr = module._malloc(modelData.length);
    module.HEAPU8.set(modelData, ptr);
    module._load_model_from_buffer(0, ptr, modelData.length);
    module._free(ptr);
    self.postMessage({ type: 'ready' });
    return;
  }

  if (type === 'separate') {
    const inPtr = module._malloc(nSamples * 4);
    module.HEAPF32.set(audioData, inPtr >> 2);
    module._separate(0, inPtr, nSamples);
    const outPtr = module._get_output_ptr();
    const result = new Float32Array(nSamples);
    result.set(module.HEAPF32.subarray(outPtr >> 2, (outPtr >> 2) + nSamples));
    module._free(inPtr);
    self.postMessage({ type: 'done', result, modelIdx }, [result.buffer]);
    return;
  }
};
