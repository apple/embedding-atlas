export function deindent(code) {
  let lines = code.split(/\r?\n/);
  let indent = null;
  for (const line of lines) {
    if (line.trim() === "") {
      // skip blank lines
      continue;
    }
    const leading = line.match(/^ */)[0].length;
    if (indent === null || leading < indent) {
      indent = leading;
    }
  }
  if (indent === null) return lines.join("\n");
  return lines.map((line) => (line.trim() === "" ? line : line.slice(indent))).join("\n");
}

export function prepareHTML(strings, ...values) {
  let out = "";
  for (let i = 0; i < strings.length; i++) {
    out += strings[i];
    if (i < values.length)
      out += values[i]
        .toString()
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
  }
  return deindent(out);
}
