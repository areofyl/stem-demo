// Web Worker — loads all 4 models into one WASM instance, runs sequentially
importScripts('stem.js');

let module = null;

self.onmessage = async function(e) {
  const { type } = e.data;

  if (type === 'init') {
    // init WASM and load all 4 models
    module = await StemModule();
    const { models } = e.data;
    for (let i = 0; i < models.length; i++) {
      const data = models[i];
      const ptr = module._malloc(data.length);
      module.HEAPU8.set(data, ptr);
      module._load_model_from_buffer(i, ptr, data.length);
      module._free(ptr);
    }
    self.postMessage({ type: 'ready' });
    return;
  }

  if (type === 'separate') {
    const { audioData, nSamples, stemNames } = e.data;

    // allocate input once
    const inPtr = module._malloc(nSamples * 4);
    module.HEAPF32.set(audioData, inPtr >> 2);

    const results = {};
    for (let i = 0; i < stemNames.length; i++) {
      module._separate(i, inPtr, nSamples);
      const outPtr = module._get_output_ptr();
      const result = new Float32Array(nSamples);
      result.set(module.HEAPF32.subarray(outPtr >> 2, (outPtr >> 2) + nSamples));
      results[stemNames[i]] = result;
      self.postMessage({ type: 'progress', done: i + 1, total: stemNames.length });
    }

    module._free(inPtr);
    self.postMessage({ type: 'done', results }, Object.values(results).map(r => r.buffer));
    return;
  }
};
