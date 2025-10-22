// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { generatePythonCommandLineDocs } from "./doc_python_cli.mjs";
import { renderPythonDocstring } from "./doc_python_docstring.mjs";
import { generateTypeScriptInterfaceDocs } from "./doc_ts_interface.mjs";

/** A markdown-it plugin that turns <!-- @doc: InterfaceName --> into documentation of the interface. */
export default function docComment(md, options) {
  function rule(state) {
    state.src = state.src.replaceAll(/<!--\s\@doc\((.*?)\):\s(.*?)\s-->/g, (_, g1, g2) => {
      let args = g1.split(",").map((x) => x.trim());
      let name = g2.trim();
      if (args[0] == "ts") {
        return generateTypeScriptInterfaceDocs(name, options.indexDts, {
          noRequired: args.indexOf("no-required") >= 0,
        });
      } else if (args[0] == "python-cli") {
        return generatePythonCommandLineDocs(name);
      } else if (args[0] == "python-docstring") {
        return renderPythonDocstring(name);
      } else {
        return "Unknown Directive";
      }
    });
  }
  md.core.ruler.before("normalize", "docComment", rule);
}
