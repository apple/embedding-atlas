<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts" module>
  import { untrack } from "svelte";

  // CodeMirror packages
  import { autocompletion, closeBrackets, closeBracketsKeymap, completionKeymap } from "@codemirror/autocomplete";
  import { defaultKeymap, history, historyKeymap, indentLess, insertTab } from "@codemirror/commands";
  import { bracketMatching, defaultHighlightStyle, indentOnInput, syntaxHighlighting } from "@codemirror/language";
  import { lintKeymap } from "@codemirror/lint";
  import { highlightSelectionMatches, searchKeymap } from "@codemirror/search";
  import { EditorState } from "@codemirror/state";
  import {
    crosshairCursor,
    drawSelection,
    dropCursor,
    EditorView,
    highlightActiveLine,
    highlightActiveLineGutter,
    highlightSpecialChars,
    keymap,
    lineNumbers,
    rectangularSelection,
    tooltips,
  } from "@codemirror/view";

  // Languages
  import { markdown } from "@codemirror/lang-markdown";
  import { PostgreSQL, sql } from "@codemirror/lang-sql";
  import { jsonSchema } from "codemirror-json-schema";
  import { json5Schema } from "codemirror-json-schema/json5";

  // Theme
  import { oneDark } from "@codemirror/theme-one-dark";

  interface JSONOptions {
    schema?: any;
  }

  interface SQLOptions {
    table?: string;
    columns?: { name: string; type: string }[];
  }

  function basicExtensions() {
    return [
      lineNumbers(),
      highlightActiveLineGutter(),
      highlightSpecialChars(),
      history(),
      drawSelection(),
      dropCursor(),
      EditorState.allowMultipleSelections.of(true),
      indentOnInput(),
      syntaxHighlighting(defaultHighlightStyle, { fallback: true }),
      bracketMatching(),
      closeBrackets(),
      autocompletion(),
      rectangularSelection(),
      crosshairCursor(),
      highlightActiveLine(),
      highlightSelectionMatches(),
      EditorView.domEventHandlers({
        keydown(event) {
          event.stopPropagation();
        },
      }),
      keymap.of([
        ...closeBracketsKeymap,
        ...defaultKeymap,
        ...searchKeymap,
        ...historyKeymap,
        ...completionKeymap,
        ...lintKeymap,
        {
          key: "Tab",
          preventDefault: true,
          run: insertTab,
        },
        {
          key: "Shift-Tab",
          preventDefault: true,
          run: indentLess,
        },
      ]),
    ];
  }

  function makeCodeMirror(
    container: HTMLElement,
    options: {
      colorScheme: "light" | "dark";
      language: string;
      initialText: string;
      json?: JSONOptions;
      sql?: SQLOptions;
      onChange?: (newText: string) => void;
    },
  ) {
    const fixedHeightEditor = EditorView.theme({
      "&.cm-editor": { height: "100%" },
      ".cm-scroller": { overflow: "auto" },
      "&.cm-focused": { outline: "none" },
      ".cm-content": {
        fontFamily: "var(--font-mono)",
      },
      ".cm-tooltip": {
        boxShadow: options.colorScheme == "light" ? "0 2px 5px rgba(0,0,0,0.2)" : "0 2px 5px rgba(0,0,0,1)",
      },
      ".cm6-json-schema-hover": { padding: "6px" },
    });

    const onChangeExtension = EditorView.updateListener.of((update) => {
      if (update.docChanged) {
        if (options.onChange) {
          const newValue = update.state.doc.toString();
          options.onChange(newValue);
        }
      }
    });

    const languageExtensions = [];

    if (options.language == "json") {
      languageExtensions.push(jsonSchema(options.json?.schema ?? {}));
    }
    if (options.language == "json5") {
      languageExtensions.push(json5Schema(options.json?.schema ?? {}));
    }
    if (options.language == "sql") {
      let table = options.sql?.table;
      let columns = options.sql?.columns;
      languageExtensions.push(
        sql({
          dialect: PostgreSQL,
          upperCaseKeywords: true,
          schema:
            table != null
              ? {
                  [table]:
                    columns?.map((x) => ({
                      label: x.name,
                      detail: x.type,
                      type: "property",
                    })) ?? [],
                }
              : undefined,
          defaultTable: table,
        }),
      );
    }
    if (options.language == "markdown") {
      languageExtensions.push(markdown());
    }

    const view = new EditorView({
      doc: options.initialText,
      parent: container,
      extensions: [
        // Basic setup
        ...basicExtensions(),

        // Tooltip parent element
        tooltips({ parent: container.parentElement! }),

        // Langauge
        ...languageExtensions,

        // Theme and styling
        ...(options.colorScheme == "dark" ? [oneDark] : []),
        fixedHeightEditor,

        // Change
        onChangeExtension,
      ],
    });

    return {
      set: (value: string) => {
        view.dispatch({
          changes: {
            from: 0,
            to: view.state.doc.length,
            insert: value,
          },
        });
      },
      destroy: () => {
        view.destroy();
      },
    };
  }
</script>

<script lang="ts">
  interface Props {
    value?: string | null;
    colorScheme?: "light" | "dark";
    language?: string;
    class?: string | null;

    // Language-specific options
    json?: {
      schema?: any;
    };

    sql?: {
      table?: string;
      columns?: { name: string; type: string }[];
    };

    onChange?: (value: string) => void;
  }

  let props: Props = $props();

  let container: HTMLDivElement;

  $effect(() => {
    let currentValue = untrack(() => props.value) ?? "";

    let editor = makeCodeMirror(container, {
      language: props.language ?? "plain",
      colorScheme: props.colorScheme ?? "light",
      initialText: untrack(() => props.value) ?? "",
      json: props.json,
      sql: props.sql,
      onChange: (v) => {
        currentValue = v;
        props.onChange?.(v);
      },
    });

    $effect.pre(() => {
      if (props.value != null && props.value != currentValue) {
        currentValue = props.value;
        editor.set(currentValue);
      }
    });

    return () => {
      editor.destroy();
    };
  });
</script>

<div class={props.class ?? ""}>
  <div
    bind:this={container}
    class="rounded-md overflow-hidden border border-slate-200 dark:border-slate-600 w-full h-full"
  ></div>
</div>
