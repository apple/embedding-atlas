// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { defaultCategoryColors } from "@embedding-atlas/component";
import { mergeUpdates } from "@embedding-atlas/utils";
import * as d3 from "d3";
import colors from "tailwindcss/colors";
export { defaultCategoryColors } from "@embedding-atlas/component";

const interpolates: Record<string, (v: number) => string> = {
  blues: d3.interpolateBlues,
  brbg: d3.interpolateBrBG,
  bugn: d3.interpolateBuGn,
  bupu: d3.interpolateBuPu,
  cividis: d3.interpolateCividis,
  cool: d3.interpolateCool,
  cubehelixdefault: d3.interpolateCubehelixDefault,
  gnbu: d3.interpolateGnBu,
  greens: d3.interpolateGreens,
  greys: d3.interpolateGreys,
  inferno: d3.interpolateInferno,
  magma: d3.interpolateMagma,
  oranges: d3.interpolateOranges,
  orrd: d3.interpolateOrRd,
  piyg: d3.interpolatePiYG,
  plasma: d3.interpolatePlasma,
  prgn: d3.interpolatePRGn,
  pubu: d3.interpolatePuBu,
  pubugn: d3.interpolatePuBuGn,
  puor: d3.interpolatePuOr,
  purd: d3.interpolatePuRd,
  purples: d3.interpolatePurples,
  rainbow: d3.interpolateRainbow,
  rdbu: d3.interpolateRdBu,
  rdgy: d3.interpolateRdGy,
  rdpu: d3.interpolateRdPu,
  rdylbu: d3.interpolateRdYlBu,
  rdylgn: d3.interpolateRdYlGn,
  reds: d3.interpolateReds,
  sinebow: d3.interpolateSinebow,
  spectral: d3.interpolateSpectral,
  turbo: d3.interpolateTurbo,
  viridis: d3.interpolateViridis,
  warm: d3.interpolateWarm,
  ylgn: d3.interpolateYlGn,
  ylgnbu: d3.interpolateYlGnBu,
  ylorbr: d3.interpolateYlOrBr,
  ylorrd: d3.interpolateYlOrRd,
};

export interface ChartTheme {
  scheme: "light" | "dark";

  /** Default interpolate for continuous color scales */
  interpolate: string | string[] | ((v: number) => string);

  /** Category color scheme */
  categoryColors: string[] | ((count: number) => string[]);

  /** Mark color */
  markColor: string;
  markColorFade: string;
  markColorGray: string;
  markColorGrayFade: string;
  ruleColor: string;

  /** Grid color */
  gridColor: string;

  /** Label color */
  labelColor: string;
  labelFontFamily: string;
  labelFontSize: number;
  labelMaxWidth: number;

  /** Border of the brush selection */
  brushBorder: string;

  /** Back border of the brush selection */
  brushBorderBack: string;

  /** Fill color of the brush selection */
  brushFill: string;
}

export type ChartThemeConfig = Partial<ChartTheme> & { light?: Partial<ChartTheme>; dark?: Partial<ChartTheme> };

function adjustForGray(hex: string, lightnessShift: number = 0): string {
  let c = d3.lab(hex);
  c.l += lightnessShift;
  c.a = 0;
  c.b = 0;
  return c.rgb().formatHex8();
}

const defaultTheme: { light: ChartTheme; dark: ChartTheme } = {
  light: {
    scheme: "light",
    interpolate: "pubugn",
    categoryColors: defaultCategoryColors,
    markColor: "#3b82f6",
    markColorFade: "#dbeafe",
    markColorGray: adjustForGray("#3b82f6", 20),
    markColorGrayFade: adjustForGray("#dbeafe"),
    ruleColor: "#000",
    gridColor: colors.slate[300],
    labelColor: colors.slate[400],
    labelFontFamily: "system-ui",
    labelFontSize: 11,
    labelMaxWidth: 80,
    brushBorder: colors.slate[500],
    brushBorderBack: "#fff",
    brushFill: "rgba(0,0,0,0.1)",
  },
  dark: {
    scheme: "dark",
    interpolate: "inferno",
    categoryColors: defaultCategoryColors,
    markColor: "#3b82f6",
    markColorFade: "#3b4d7f",
    markColorGray: adjustForGray("#3b82f6", -20),
    markColorGrayFade: adjustForGray("#1f398a"),
    ruleColor: "#fff",
    gridColor: colors.slate[700],
    labelColor: colors.slate[500],
    labelFontFamily: "system-ui",
    labelFontSize: 11,
    labelMaxWidth: 80,
    brushBorder: colors.slate[400],
    brushBorderBack: "#000",
    brushFill: "rgba(255,255,255,0.1)",
  },
};

export function resolveChartTheme(scheme: "light" | "dark", config?: ChartThemeConfig): ChartTheme {
  let theme = defaultTheme[scheme];
  if (config != undefined) {
    theme = mergeUpdates(theme, mergeUpdates(config, { light: undefined, dark: undefined }) ?? config) ?? theme;
    if (config[scheme] != undefined) {
      theme = mergeUpdates(theme, config[scheme]) ?? theme;
    }
  }
  return theme;
}

export function resolveInterpolate(value: string | string[] | ((v: number) => string)): (v: number) => string {
  if (typeof value == "string") {
    return interpolates[value.toLowerCase()] ?? d3.interpolateTurbo;
  } else if (typeof value == "function") {
    return value;
  } else {
    return d3.interpolateRgbBasis(value);
  }
}

export function defaultOrdinalColors(count: number): string[] {
  let result: string[] = [];
  for (let i = 0; i < count; i++) {
    result.push(d3.interpolateTurbo((i + 0.5) / count));
  }
  return result;
}
