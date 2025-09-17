import { readFileSync } from "fs";
import * as ts from "typescript";

import { prepareHTML } from "./utils.mjs";

const printer = ts.createPrinter({
  newLine: ts.NewLineKind.LineFeed,
  removeComments: true,
  omitTrailingSemicolon: true,
});

function extractMembers(node, sourceFile) {
  let result = [];
  for (let member of node.members) {
    if (ts.isPropertySignature(member) && member.type) {
      let name = member.name.getText(sourceFile);
      ts.setEmitFlags(member.type, ts.EmitFlags.SingleLine);
      let memberType = printer.printNode(ts.EmitHint.Unspecified, member.type, sourceFile);
      let item = {
        name: name,
        type: memberType,
        docs: member.jsDoc?.[0]?.comment,
        isRequired: member.questionToken == undefined,
      };
      if (ts.isTypeLiteralNode(member.type)) {
        item.members = extractMembers(member.type);
      }
      result.push(item);
    }
  }
  return result;
}

function extract(dtsContents) {
  let sourceFile = ts.createSourceFile("index", dtsContents, ts.ScriptTarget.ESNext, true, ts.ScriptKind.TS);

  let result = [];

  ts.forEachChild(sourceFile, (node) => {
    if (ts.isInterfaceDeclaration(node)) {
      let item = {
        kind: "InterfaceDeclaration",
        name: node.name.getText(sourceFile),
        members: extractMembers(node, sourceFile),
      };
      result.push(item);
    }
  });
  return result;
}

function renderType(type) {
  while (true) {
    let n = type.replaceAll(/_\d+($|(?=\ ))/g, ""); // remove _2, _3, ... suffix
    if (type == n) {
      break;
    }
    type = n;
  }
  return type.replaceAll("\n", "");
}

const loadFileCache = new Map();
function loadFile(path) {
  if (loadFileCache.has(path)) {
    return loadFileCache.get(path);
  }
  try {
    let data = extract(readFileSync(path, { encoding: "utf-8" }));
    loadFileCache.set(path, data);
    return data;
  } catch (e) {
    return [];
  }
}

export function generateTypeScriptInterfaceDocs(interfaceName, indexDts, options = {}) {
  const data = loadFile(indexDts);
  const noRequired = options.noRequired ?? false;

  let node = data.filter((x) => x.name == interfaceName)[0];

  if (node == null) {
    return "";
  }

  let chunks = [];

  for (let member of node.members) {
    let id = node.name + "." + member.name;
    chunks.push(
      prepareHTML`
        <h3 id="${id}" class="doc-property">
          ${member.name}
          <code>${renderType(member.type)}</code>
          <a class="header-anchor" href="#${id}" aria-label="member ${member.name} of ${node.name}">​</a>
        </h3>
      `,
    );
    if (member.isRequired && !noRequired) {
      chunks.push("Required. " + (member.docs ?? ""));
    } else if (member.docs != null) {
      chunks.push(member.docs);
    }

    if (member.members) {
      for (let subMember of member.members) {
        let subId = id + "." + subMember.name;
        chunks.push(
          prepareHTML`
            <h4 id="${subId}" class="doc-property">
              ${subMember.name}
              <code>${renderType(subMember.type)}</code>
              <a class="header-anchor" href="#${subId}" aria-label="member ${subMember.name} of ${node.name}">​</a>
            </h4>
          `,
        );
        if (subMember.isRequired && !noRequired) {
          chunks.push("Required. " + (subMember.docs ?? ""));
        } else if (subMember.docs != null) {
          chunks.push(subMember.docs);
        }
      }
    }
  }
  return "\n\n" + chunks.filter((x) => x != "").join("\n\n") + "\n\n";
}
