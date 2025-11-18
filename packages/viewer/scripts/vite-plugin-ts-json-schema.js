// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { readFileSync } from "fs";
import { resolve } from "path";
import { createGenerator } from "ts-json-schema-generator";

/**
 * Vite plugin that converts TypeScript interfaces to JSON schemas
 * @param {Object} options - Plugin options
 * @param {string} [options.tsconfig='tsconfig.json'] - Path to tsconfig.json
 * @param {Object} [options.schemaConfig={}] - Additional ts-json-schema-generator config
 * @returns {Object} Vite plugin
 */
function tsJsonSchemaPlugin(options = {}) {
  let { tsconfig = "tsconfig.json", schemaConfig = {} } = options;

  function checkId(id) {
    return id.endsWith("&json-schema");
  }

  return {
    name: "ts-json-schema",

    resolveId(id) {
      // Check if this is a request for a JSON schema
      if (checkId(id)) {
        return id; // Mark this as a virtual module
      }
      return null;
    },

    load(id) {
      if (!checkId(id)) {
        return null;
      }

      // Parse the query parameters
      let url = new URL(id, "file://");
      let typeName = url.searchParams.get("type");

      if (!typeName) {
        return null;
      }

      try {
        // Get the actual file path (remove query parameters)
        let filePath = id.split("?")[0];
        let absolutePath = resolve(filePath);

        // Verify the file exists
        readFileSync(absolutePath, "utf8");

        // Create the schema generator config
        let config = {
          path: absolutePath,
          tsconfig: resolve(tsconfig),
          type: typeName,
          skipTypeCheck: true, // For better performance
          ...schemaConfig,
        };

        // Generate the schema
        let generator = createGenerator(config);
        let schema = generator.createSchema(typeName);

        // Return the schema as a JavaScript module
        return `export default ${JSON.stringify(schema, null, 2)};`;
      } catch (error) {
        // Provide helpful error messages
        let errorMessage = error instanceof Error ? error.message : String(error);

        this.error(`Failed to generate JSON schema for interface "${typeName}" in file "${id}": ${errorMessage}`);
      }
    },

    // Handle HMR (Hot Module Replacement)
    handleHotUpdate({ file, server }) {
      // If a TypeScript file changes, invalidate any related schema modules
      if (file.endsWith(".ts") || file.endsWith(".tsx")) {
        let modules = server.moduleGraph.getModulesByFile(file);
        if (modules) {
          for (let module of modules) {
            // Find any virtual modules that depend on this file
            for (let importer of module.importers) {
              if (importer.id && importer.id.includes("?jsonSchema=")) {
                server.reloadModule(importer);
              }
            }
          }
        }
      }
    },
  };
}

export { tsJsonSchemaPlugin };
