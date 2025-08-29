import Module from "./runtime.js";

async function initialize() {
  const module = await Module();
  const ret = {
    memory_allocate: module._memory_allocate,
    memory_free: module._memory_free,
    memory_size: () => module.HEAP32.length / 4,
    u8_array: (ptr, length) => module.HEAPU8.subarray(ptr, ptr + length),
    f32_array: (ptr, length) => module.HEAPF32.subarray(ptr / 4, ptr / 4 + length),
    i32_array: (ptr, length) => module.HEAP32.subarray(ptr / 4, ptr / 4 + length),
  };
  for (let key in module) {
    if (key.startsWith("_")) {
      ret[key.substring(1)] = module[key];
    }
  }
  return ret;
}

let initializePromise = initialize();

function set_options(ctx, ptr, setters, options) {
  const buffer_length = 64;
  const buffer = ctx.memory_allocate(buffer_length * 2);
  const str_buffer_1 = buffer;
  const str_buffer_2 = buffer + buffer_length;

  function set_str_buffer(buf, value) {
    const encoded = new TextEncoder().encode(value);
    if (encoded.length + 1 > buffer_length) {
      throw new Error("invalid parameter " + value);
    }
    const array = ctx.u8_array(buf, 64);
    array.fill(0);
    array.set(encoded);
    array[array.length - 1] = 0;
  }

  for (const name in options) {
    // Convert camel case to snake case (expected by the C++ code)
    let name_snake_case = name.replace(/([a-z0-9])([A-Z])/g, "$1_$2").toLowerCase();
    set_str_buffer(str_buffer_1, name_snake_case);
    // Set the value
    const value = options[name];
    if (typeof value == "number") {
      const r = setters.number(ptr, str_buffer_1, value);
      if (r != 0) {
        throw new Error("invalid parameter " + name);
      }
    } else if (typeof value == "string") {
      set_str_buffer(str_buffer_2, value);
      const r = setters.string(ptr, str_buffer_1, str_buffer_2);
      if (r != 0) {
        throw new Error("invalid parameter " + name);
      }
    } else {
      throw new Error("invalid parameter " + name);
    }
  }

  ctx.memory_free(str_buffer_1);
}

export async function createUMAP(count, input_dim, output_dim, data, options = {}) {
  const ctx = await initializePromise;

  const ptr_input = ctx.memory_allocate(count * input_dim * 4);
  const ptr_output = ctx.memory_allocate(count * output_dim * 4);

  const ptr_options = ctx.umap_options_create();
  set_options(ctx, ptr_options, { number: ctx.umap_options_number, string: ctx.umap_options_string }, options);

  ctx.f32_array(ptr_input, count * input_dim).set(data, 0);

  let ptr_umap = ctx.umap_context_create_f32(count, input_dim, output_dim, ptr_input, ptr_output, ptr_options);

  ctx.umap_options_destroy(ptr_options);

  function assertValid() {
    if (ptr_umap == null) {
      throw new Error("use after destroy");
    }
  }

  return {
    get epoch() {
      assertValid();
      return ctx.umap_context_epoch(ptr_umap);
    },
    get nEpochs() {
      assertValid();
      return ctx.umap_context_n_epochs(ptr_umap);
    },
    get count() {
      return count;
    },
    get outputDim() {
      return output_dim;
    },
    get inputDim() {
      return input_dim;
    },
    get embedding() {
      assertValid();
      return ctx.f32_array(ptr_output, count * output_dim);
    },
    run(epoch_limit) {
      assertValid();
      ctx.umap_context_run(ptr_umap, epoch_limit ?? 0);
    },
    destroy() {
      if (ptr_umap == null) {
        return;
      }
      ctx.umap_context_destroy(ptr_umap);
      ctx.memory_free(ptr_input);
      ctx.memory_free(ptr_output);
      ptr_umap = null;
    },
  };
}

export async function createKNN(count, input_dim, data, options) {
  const ctx = await initializePromise;
  const ptr_options = ctx.knn_options_create();
  set_options(ctx, ptr_options, { number: ctx.knn_options_number, string: ctx.knn_options_string }, options);

  const ptr_input = ctx.memory_allocate(count * input_dim * 4);
  ctx.f32_array(ptr_input, count * input_dim).set(data, 0);

  let ptr_knn = ctx.knn_context_create_f32(count, input_dim, ptr_input, ptr_options);

  ctx.knn_options_destroy(ptr_options);

  let buffer_capacity = 10;
  let ptr_buffer = ctx.memory_allocate(10 * 4);
  const ptr_q = ctx.memory_allocate(input_dim * 4);

  function get_buffers(k) {
    let required_capacity = k * 2;
    if (buffer_capacity < required_capacity) {
      ctx.memory_free(ptr_buffer);
      ptr_buffer = ctx.memory_allocate(required_capacity * 4 * 2);
      buffer_capacity = required_capacity * 4 * 2;
    }
    return [ptr_buffer, ptr_buffer + k * 4];
  }

  return {
    queryByIndex(index, k) {
      let [ptr_i, ptr_d] = get_buffers(k);
      let rk = ctx.knn_context_query_by_index(ptr_knn, index, k, ptr_i, ptr_d);
      return {
        indices: ctx.i32_array(ptr_i, rk).slice(),
        distances: ctx.f32_array(ptr_d, rk).slice(),
      };
    },
    queryByVector(data, k) {
      let [ptr_i, ptr_d] = get_buffers(k);
      ctx.f32_array(ptr_q, input_dim).set(data);
      let rk = ctx.knn_context_query_by_vector(ptr_knn, ptr_q, k, ptr_i, ptr_d);
      return {
        indices: ctx.i32_array(ptr_i, rk).slice(),
        distances: ctx.f32_array(ptr_d, rk).slice(),
      };
    },
    destroy() {
      if (ptr_knn == null) {
        return;
      }
      ctx.knn_context_destroy(ptr_knn);
      ctx.memory_free(ptr_input);
      ctx.memory_free(ptr_buffer);
      ctx.memory_free(ptr_q);
      ptr_knn = null;
    },
  };
}

export async function memorySize() {
  let ctx = await initializePromise;
  return ctx.memory_size();
}
