import { execSync } from "child_process";
import { resolve } from "path";

import { deindent, prepareHTML } from "./utils.mjs";

function extract(entrypoint) {
  let [module, func] = entrypoint.split(":");
  let code = deindent(`
    import json
    from ${module} import ${func} as main
    print(json.dumps([p.to_info_dict() for p in main.params]))
  `);
  let helpCommand = `uv run python -c "${code}"`;
  let workDir = resolve("../../packages/backend");
  let json = execSync(helpCommand, { cwd: workDir, encoding: "utf-8" });
  return JSON.parse(json);
}

function generate(args) {
  let lines = [];
  for (let arg of args) {
    if (arg.param_type_name != "option") {
      continue;
    }
    let id = arg.name;
    let opts = [...arg.opts, ...arg.secondary_opts].join(" / ");
    lines.push(
      prepareHTML`
        <h4 id=${id}>
          ${opts} <code>${arg.type.name}</code>
          <a class="header-anchor" href="#${id}" aria-label="option ${opts}">​</a>
        </h4>
      `,
    );
    lines.push(arg.help);
  }
  return lines.join("\n\n");
}

export function generatePythonCommandLineDocs(name) {
  let data = extract(name);
  return generate(data);
}
