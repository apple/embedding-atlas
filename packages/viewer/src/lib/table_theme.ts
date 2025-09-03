// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

/**
 * This file has inline comments with tailwind variables such that
 * the corresponding color variables are available for use.
 * DO NOT REMOVE THE COMMENTS!
 */

export const tableTheme = {
  fontSize: "13px",
  fontFamily: "system-ui",
  light: {
    primaryBackgroundColor: "white",
    secondaryBackgroundColor: "var(--color-slate-100)", // bg-slate-100
    primaryTextColor: "var(--color-slate-500)", // text-slate-500
    secondaryTextColor: "var(--color-slate-400)", // text-slate-400
    tertiaryTextColor: "var(--color-slate-300)", // text-slate-300
    scrollbarPillColor: "var(--color-slate-400)", // bg-slate-400
    scrollbarLabelBackgroundColor: "white",
    rowScrollToColor: "var(--color-blue-200)", // bg-blue-200
    rowHoverColor: "var(--color-blue-100)", // bg-blue-100
  },
  dark: {
    primaryBackgroundColor: "var(--color-slate-900)", // bg-slate-900
    secondaryBackgroundColor: "var(--color-slate-800)", // bg-slate-800
    primaryTextColor: "var(--color-slate-400)", // bg-slate-400
    secondaryTextColor: "var(--color-slate-500)", // bg-slate-500
    tertiaryTextColor: "var(--color-slate-600)", // bg-slate-600
    scrollbarPillColor: "var(--color-slate-500)", // bg-slate-500
    scrollbarLabelBackgroundColor: "var(--color-slate-900)", // bg-slate-900
    rowScrollToColor: "var(--color-blue-900)", // bg-slate-900
    rowHoverColor: "var(--color-blue-950)", // bg-slate-950
  },
};
