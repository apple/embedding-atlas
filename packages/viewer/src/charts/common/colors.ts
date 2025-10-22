// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { lab } from "d3-color";
import { interpolateInferno, interpolateTurbo, interpolateYlGnBu } from "d3-scale-chromatic";
import colors from "tailwindcss/colors";
export { defaultCategoryColors } from "@embedding-atlas/component";

function adjustForGray(hex: string, lightnessShift: number = 0): string {
  let c = lab(hex);
  c.l += lightnessShift;
  c.a = 0;
  c.b = 0;
  return c.rgb().formatHex8();
}

export const chartColors = {
  light: {
    scheme: "light",
    continuousColorScheme: "YlGnBu",
    continuousColorSchemeAtZero: interpolateYlGnBu(0),
    markColor: "#3b82f6",
    markColorFade: "#dbeafe",
    markColorGray: adjustForGray("#3b82f6", 20),
    markColorGrayFade: adjustForGray("#dbeafe"),
    gridColor: colors.slate[300],
    labelColor: colors.slate[400],
    titleColor: colors.slate[400],
    brushBorder: colors.slate[500],
    brushBorderBack: "#fff",
    brushFill: "rgba(0,0,0,0.1)",
  },
  dark: {
    scheme: "dark",
    continuousColorScheme: "Inferno",
    continuousColorSchemeAtZero: interpolateInferno(0),
    markColor: "#3b82f6",
    markColorFade: "#3b4d7f",
    markColorGray: adjustForGray("#3b82f6", -20),
    markColorGrayFade: adjustForGray("#1f398a"),
    gridColor: colors.slate[700],
    labelColor: colors.slate[500],
    titleColor: colors.slate[500],
    brushBorder: colors.slate[400],
    brushBorderBack: "#000",
    brushFill: "rgba(255,255,255,0.1)",
  },
};

export function defaultOrdinalColors(count: number): string[] {
  let result: string[] = [];
  for (let i = 0; i < count; i++) {
    result.push(interpolateTurbo((i + 0.5) / count));
  }
  return result;
}
