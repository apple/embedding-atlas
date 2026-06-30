// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

export interface IProgram {
  program: WebGLProgram;
  uniforms: { [key: string]: WebGLUniformLocation };
}

export function webglProgram(
  state: {
    program?: WebGLProgram;
    uniforms?: { [key: string]: WebGLUniformLocation };
    vsSource?: string;
    fsSource?: string;
    destroy?: () => void;
  },
  gl: WebGL2RenderingContext,
  vsSource: string,
  fsSource: string,
): IProgram {
  if (state.program == null || state.vsSource != vsSource || state.fsSource != fsSource) {
    if (state.destroy) {
      state.destroy();
    }
    let vs = compileShader(gl, gl.VERTEX_SHADER, vsSource);
    let fs = compileShader(gl, gl.FRAGMENT_SHADER, fsSource);
    let program = gl.createProgram()!;
    gl.attachShader(program, vs);
    gl.attachShader(program, fs);
    gl.linkProgram(program);
    let linkStatus = gl.getProgramParameter(program, gl.LINK_STATUS);
    if (!linkStatus) {
      var log = gl.getProgramInfoLog(program);
      throw new Error(`failed to link program: ${log}, vertex source: ${vsSource}, fragment source: ${fsSource}`);
    }
    state.program = program;
    state.vsSource = vsSource;
    state.fsSource = fsSource;
    state.destroy = () => {
      gl.deleteProgram(program);
      gl.deleteShader(vs);
      gl.deleteShader(fs);
    };
    state.uniforms = {};
    for (let m of (vsSource + fsSource).matchAll(/uniform +[0-9a-zA-Z_]+ +([0-9a-zA-Z_]+) *(;|\[)/g)) {
      let name = m[1];
      state.uniforms[name] = gl.getUniformLocation(program, name)!;
    }
  }
  return { program: state.program, uniforms: state.uniforms ?? {} };
}

function compileShader(gl: WebGL2RenderingContext, type: number, source: string): WebGLShader {
  let shader = gl.createShader(type)!;
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  let compileStatus = gl.getShaderParameter(shader, gl.COMPILE_STATUS);
  if (!compileStatus) {
    var log = gl.getShaderInfoLog(shader);
    throw new Error(`failed to compile shader: ${log}, source: ${source}`);
  }
  return shader;
}

export function webglBuffer(
  state: {
    buffer?: WebGLBuffer;
    data?: ArrayBufferView | number[];
    destroy?: () => void;
  },
  gl: WebGL2RenderingContext,
  data: ArrayBufferView | number[],
  type?: "f32" | "i32" | "u32" | "i16" | "u16" | "i8" | "u8",
): WebGLBuffer {
  if (state.buffer == null) {
    let buffer = gl.createBuffer()!;
    state.buffer = buffer;
    state.destroy = () => {
      gl.deleteBuffer(buffer);
    };
  }
  if (state.data !== data) {
    state.data = data;
    gl.bindBuffer(gl.ARRAY_BUFFER, state.buffer);
    if (data instanceof Array) {
      switch (type ?? "f32") {
        case "f32":
          gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(data), gl.STATIC_DRAW);
          break;
        case "i32":
          gl.bufferData(gl.ARRAY_BUFFER, new Int32Array(data), gl.STATIC_DRAW);
          break;
        case "u32":
          gl.bufferData(gl.ARRAY_BUFFER, new Uint32Array(data), gl.STATIC_DRAW);
          break;
        case "i16":
          gl.bufferData(gl.ARRAY_BUFFER, new Int16Array(data), gl.STATIC_DRAW);
          break;
        case "u16":
          gl.bufferData(gl.ARRAY_BUFFER, new Uint16Array(data), gl.STATIC_DRAW);
          break;
        case "i8":
          gl.bufferData(gl.ARRAY_BUFFER, new Int8Array(data), gl.STATIC_DRAW);
          break;
        case "u8":
          gl.bufferData(gl.ARRAY_BUFFER, new Uint8Array(data), gl.STATIC_DRAW);
          break;
        default:
          throw new Error("invalid type");
      }
    } else {
      gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
  }
  return state.buffer;
}

/** A cached ELEMENT_ARRAY_BUFFER of u32 indices (used for strided 3D downsampling). */
export function webglElementBuffer(
  state: { buffer?: WebGLBuffer; data?: Uint32Array; destroy?: () => void },
  gl: WebGL2RenderingContext,
  data: Uint32Array,
): WebGLBuffer {
  if (state.buffer == null) {
    let buffer = gl.createBuffer()!;
    state.buffer = buffer;
    state.destroy = () => {
      gl.deleteBuffer(buffer);
    };
  }
  if (state.data !== data) {
    state.data = data;
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, state.buffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, data, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
  }
  return state.buffer;
}

export interface IFramebuffer {
  framebuffer: WebGLFramebuffer;
  texture: WebGLTexture;
  width: number;
  height: number;
}

function setTextureImage(
  gl: WebGL2RenderingContext,
  width: number,
  height: number,
  channels: 1 | 2 | 3 | 4,
  type: "u8" | "u16" | "u32" | "f32",
) {
  const mapping = {
    u8: {
      1: [gl.R8, gl.RED, gl.UNSIGNED_BYTE],
      2: [gl.RG8, gl.RG, gl.UNSIGNED_BYTE],
      3: [gl.RGB8, gl.RGB, gl.UNSIGNED_BYTE],
      4: [gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE],
    },
    u16: {
      1: [gl.R8, gl.RED, gl.UNSIGNED_SHORT],
      2: [gl.RG8, gl.RG, gl.UNSIGNED_SHORT],
      3: [gl.RGB8, gl.RGB, gl.UNSIGNED_SHORT],
      4: [gl.RGBA8, gl.RGBA, gl.UNSIGNED_SHORT],
    },
    u32: {
      1: [gl.R8, gl.RED, gl.UNSIGNED_INT],
      2: [gl.RG8, gl.RG, gl.UNSIGNED_INT],
      3: [gl.RGB8, gl.RGB, gl.UNSIGNED_INT],
      4: [gl.RGBA8, gl.RGBA, gl.UNSIGNED_INT],
    },
    f32: {
      1: [gl.R32F, gl.RED, gl.FLOAT],
      2: [gl.RG32F, gl.RG, gl.FLOAT],
      3: [gl.RGB32F, gl.RGB, gl.FLOAT],
      4: [gl.RGBA32F, gl.RGBA, gl.FLOAT],
    },
  };
  let [internalFormat, format, dtype] = mapping[type][channels];
  gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, width, height, 0, format, dtype, null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
}

export function webglFramebuffer(
  state: {
    framebuffer?: WebGLFramebuffer;
    texture?: WebGLTexture;
    cacheKey?: string;
    destroy?: () => void;
  },
  gl: WebGL2RenderingContext,
  width: number,
  height: number,
  channels: 1 | 2 | 3 | 4,
  type: "u8" | "u16" | "u32" | "f32",
): IFramebuffer {
  if (state.framebuffer == null || state.texture == null) {
    let framebuffer = gl.createFramebuffer()!;
    let texture = gl.createTexture()!;

    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
    gl.bindTexture(gl.TEXTURE_2D, null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    state.framebuffer = framebuffer;
    state.texture = texture;
    state.destroy = () => {
      gl.deleteFramebuffer(framebuffer);
      gl.deleteTexture(texture);
    };
  }
  let cacheKey = `${width},${height},${channels},${type}`;
  if (state.cacheKey != cacheKey) {
    state.cacheKey = cacheKey;
    gl.bindTexture(gl.TEXTURE_2D, state.texture);
    setTextureImage(gl, width, height, channels, type);
    gl.bindTexture(gl.TEXTURE_2D, null);
  }
  return {
    framebuffer: state.framebuffer,
    texture: state.texture,
    width: width,
    height: height,
  };
}
