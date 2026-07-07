import type { BuiltinChartSpec, EmbeddingAtlasState } from "embedding-atlas";

import { encode } from "./utils.js";

interface Example {
  title: string;
  details: string;
  image: string;
  data: string;
  settings?: any;
  state?: Omit<EmbeddingAtlasState, "charts"> & { charts: Record<string, BuiltinChartSpec> };
}

const examples: Record<string, Example[]> = {
  embedding: [
    {
      title: "Wine Reviews",
      details: "Data from Hugging Face: spawn99/wine-reviews",
      image: "/assets/examples/wine-reviews-${colorscheme}.jpg",
      data: "example://wine-reviews",
      settings: {
        text: "description",
        embedding: { precomputed: { x: "projection_x", y: "projection_y", neighbors: "neighbors" } },
      },
      state: {
        version: "0.21.0",
        charts: {
          1: {
            type: "embedding",
            title: "Embedding",
            data: { x: "projection_x", y: "projection_y", text: "description", category: "country" },
          },
          2: { type: "predicates", title: "SQL Predicates" },
          3: {
            title: "points vs. price",
            layers: [
              {
                mark: "rect",
                filter: "$filter",
                width: 1,
                style: { fillColor: "$ruleColor" },
                encoding: {
                  x: { field: "price" },
                  y1: { aggregate: "min", field: "points" },
                  y2: { aggregate: "max", field: "points" },
                },
              },
              {
                mark: "rect",
                filter: "$filter",
                width: { gap: 1, clampToRatio: 0.1 },
                encoding: {
                  x: { field: "price" },
                  y1: { aggregate: "quantile", quantile: 0.25, field: "points" },
                  y2: { aggregate: "quantile", quantile: 0.75, field: "points" },
                },
              },
              {
                mark: "rect",
                filter: "$filter",
                height: 1,
                width: { gap: 1, clampToRatio: 0.1 },
                style: { fillColor: "$ruleColor" },
                encoding: { x: { field: "price" }, y: { aggregate: "median", field: "points" } },
              },
            ],
            selection: { brush: { encoding: "x" } },
            axis: { y: { title: "points" } },
            widgets: [
              { type: "scale.type", channel: "x" },
              { type: "scale.type", channel: "y" },
            ],
          },
          4: {
            title: "price by country",
            layers: [
              {
                mark: "bar",
                style: { fillColor: "$markColorFade" },
                encoding: { x: { field: "price" }, y: { aggregate: "count" } },
              },
              {
                mark: "bar",
                filter: "$filter",
                encoding: { x: { field: "price" }, y: { aggregate: "count" }, color: { field: "country" } },
              },
            ],
            selection: { brush: { encoding: "x" } },
            widgets: [
              { type: "scale.type", channel: "x" },
              { type: "encoding.normalize", attribute: "y", layer: [0, 1], options: ["x"] },
            ],
          },
          5: {
            type: "instances",
            title: "Wines",
            viewMode: "cards",
            cardTemplate:
              '<div class="wine-card">\n  <style>\n    .wine-card {\n      --bg: #fdfcf9;\n      --title: #1a1a2e;\n      --muted: #6b6358;\n      --accent: #7a1f2e;\n      --badge-bg: #7a1f2e;\n      --badge-fg: #fdfcf9;\n      --price-fg: #2d5016;\n      --desc: #3a3a3a;\n      --tag-bg: #f3ede0;\n      --tag-fg: #5a4a32;\n      padding: 8px 10px;\n      background: var(--bg);\n      font-family: ui-sans-serif, system-ui, -apple-system, sans-serif;\n      display: flex;\n      flex-direction: column;\n      gap: 5px;\n    }\n    .wine-card:is(.dark *) {\n      --bg: #1f1d1a;\n      --title: #f5f1e8;\n      --muted: #a89f92;\n      --accent: #d97a8a;\n      --badge-bg: #d97a8a;\n      --badge-fg: #1a1a1a;\n      --price-fg: #9bc26b;\n      --desc: #d4cfc4;\n      --tag-bg: #2e2a26;\n      --tag-fg: #c2b8a6;\n    }\n    .wine-card .header {\n      display: flex;\n      justify-content: space-between;\n      align-items: flex-start;\n      gap: 8px;\n    }\n    .wine-card .title-block { flex: 1; min-width: 0; }\n    .wine-card .title {\n      font-size: 13px;\n      font-weight: 600;\n      line-height: 1.25;\n      color: var(--title);\n    }\n    .wine-card .location {\n      font-size: 10px;\n      color: var(--muted);\n      letter-spacing: 0.3px;\n      text-transform: uppercase;\n      margin-top: 2px;\n    }\n    .wine-card .rating {\n      display: flex;\n      align-items: baseline;\n      gap: 2px;\n      padding: 2px 7px;\n      background: var(--badge-bg);\n      color: var(--badge-fg);\n      border-radius: 9px;\n      font-weight: 700;\n      flex-shrink: 0;\n      height: fit-content;\n    }\n    .wine-card .rating .pts-num { font-size: 12px; line-height: 1; }\n    .wine-card .rating .pts-lbl { font-size: 8px; opacity: 0.8; letter-spacing: 0.3px; }\n    .wine-card .meta {\n      display: flex;\n      flex-wrap: wrap;\n      gap: 4px;\n      align-items: center;\n      font-size: 10.5px;\n    }\n    .wine-card .tag {\n      padding: 1px 6px;\n      background: var(--tag-bg);\n      color: var(--tag-fg);\n      border-radius: 7px;\n      font-weight: 500;\n    }\n    .wine-card .price {\n      font-size: 11.5px;\n      font-weight: 700;\n      color: var(--price-fg);\n      margin-left: auto;\n    }\n    .wine-card .description {\n      font-size: 11.5px;\n      line-height: 1.4;\n      color: var(--desc);\n      font-style: italic;\n      border-left: 2px solid var(--accent);\n      padding-left: 7px;\n    }\n  </style>\n  <div class="header">\n    <div class="title-block">\n      <div class="title">{{ title | default: "Untitled" }}</div>\n      {% if province or country %}<div class="location">{% if province %}{{ province }}{% if country %} · {% endif %}{% endif %}{% if country %}{{ country }}{% endif %}</div>{% endif %}\n    </div>\n    {% if points %}<div class="rating"><span class="pts-num">{{ points }}</span><span class="pts-lbl">PTS</span></div>{% endif %}\n  </div>\n  <div class="meta">\n    {% if variety %}<span class="tag">{{ variety }}</span>{% endif %}\n    {% if designation %}<span class="tag">{{ designation }}</span>{% endif %}\n    {% if price %}<span class="price">${{ price }}</span>{% endif %}\n  </div>\n  {% if description %}<div class="description">{{ description }}</div>{% endif %}\n</div>',
            columns: ["title", "description", "points", "price", "variety", "designation", "country", "province"],
            sort: [
              {
                column: "points",
                direction: "descending",
              },
            ],
          },
          6: { type: "count-plot", title: "country", data: { field: "country" } },
          7: { type: "count-plot", title: "province", data: { field: "province" } },
          8: {
            title: "points",
            layers: [
              {
                mark: "bar",
                style: { fillColor: "$markColorFade" },
                encoding: { x: { field: "points" }, y: { aggregate: "count" } },
              },
              { mark: "bar", filter: "$filter", encoding: { x: { field: "points" }, y: { aggregate: "count" } } },
            ],
            selection: { brush: { encoding: "x" } },
            widgets: [
              { type: "scale.type", channel: "x" },
              { type: "encoding.normalize", attribute: "y", layer: [0, 1], options: ["x"] },
            ],
          },
          9: {
            title: "price",
            layers: [
              {
                mark: "bar",
                style: { fillColor: "$markColorFade" },
                encoding: { x: { field: "price" }, y: { aggregate: "count" } },
              },
              { mark: "bar", filter: "$filter", encoding: { x: { field: "price" }, y: { aggregate: "count" } } },
            ],
            selection: { brush: { encoding: "x" } },
            widgets: [
              { type: "scale.type", channel: "x" },
              { type: "encoding.normalize", attribute: "y", layer: [0, 1], options: ["x"] },
            ],
          },
          10: { type: "count-plot", title: "variety", data: { field: "variety" } },
        },
      },
    },
    {
      title: "Visualization Publications",
      details: "Data from vispubdata.org",
      image: "/assets/examples/vispubdata-${colorscheme}.jpg",
      data: "example://vispubdata",
      settings: {
        text: "Abstract",
        embedding: { precomputed: { x: "projection_x", y: "projection_y", neighbors: "neighbors" } },
      },
      state: {
        version: "0.21.0",
        charts: {
          1: {
            type: "embedding",
            title: "Embedding",
            data: { x: "projection_x", y: "projection_y", text: "Abstract", category: "Conference" },
          },
          2: { type: "predicates", title: "SQL Predicates" },
          3: { type: "instances", title: "Table" },
          4: {
            title: "Year",
            layers: [
              {
                mark: "bar",
                style: { fillColor: "$markColorFade" },
                encoding: { x: { field: "Year" }, y: { aggregate: "count" } },
              },
              { mark: "bar", filter: "$filter", encoding: { x: { field: "Year" }, y: { aggregate: "count" } } },
            ],
            selection: { brush: { encoding: "x" } },
            widgets: [
              { type: "scale.type", channel: "x" },
              { type: "encoding.normalize", attribute: "y", layer: [0, 1], options: ["x"] },
            ],
          },
          5: { type: "count-plot", title: "Conference", data: { field: "Conference" } },
          6: { type: "count-plot", title: "PaperType", data: { field: "PaperType" } },
          7: { type: "count-plot", title: "Award", data: { field: "Award" } },
          8: { type: "count-plot", title: "AuthorNames_Deduped", data: { field: "AuthorNames_Deduped", isList: true } },
          9: { type: "count-plot", title: "AuthorAffiliation", data: { field: "AuthorAffiliation", isList: true } },
          10: { type: "count-plot", title: "AuthorKeywords", data: { field: "AuthorKeywords", isList: true } },
          11: {
            title: "AminerCitationCount",
            layers: [
              {
                mark: "bar",
                style: { fillColor: "$markColorFade" },
                encoding: { x: { field: "AminerCitationCount" }, y: { aggregate: "count" } },
              },
              {
                mark: "bar",
                filter: "$filter",
                encoding: { x: { field: "AminerCitationCount" }, y: { aggregate: "count" } },
              },
            ],
            selection: { brush: { encoding: "x" } },
            widgets: [
              { type: "scale.type", channel: "x" },
              { type: "encoding.normalize", attribute: "y", layer: [0, 1], options: ["x"] },
            ],
          },
          12: {
            title: "CitationCount_CrossRef",
            layers: [
              {
                mark: "bar",
                style: { fillColor: "$markColorFade" },
                encoding: { x: { field: "CitationCount_CrossRef" }, y: { aggregate: "count" } },
              },
              {
                mark: "bar",
                filter: "$filter",
                encoding: { x: { field: "CitationCount_CrossRef" }, y: { aggregate: "count" } },
              },
            ],
            selection: { brush: { encoding: "x" } },
            widgets: [
              { type: "scale.type", channel: "x" },
              { type: "encoding.normalize", attribute: "y", layer: [0, 1], options: ["x"] },
            ],
          },
          13: {
            title: "PubsCited_CrossRef",
            layers: [
              {
                mark: "bar",
                style: { fillColor: "$markColorFade" },
                encoding: { x: { field: "PubsCited_CrossRef" }, y: { aggregate: "count" } },
              },
              {
                mark: "bar",
                filter: "$filter",
                encoding: { x: { field: "PubsCited_CrossRef" }, y: { aggregate: "count" } },
              },
            ],
            selection: { brush: { encoding: "x" } },
            widgets: [
              { type: "scale.type", channel: "x" },
              { type: "encoding.normalize", attribute: "y", layer: [0, 1], options: ["x"] },
            ],
          },
        },
      },
    },
    {
      title: "MedMCQA",
      details: "Data from Hugging Face: openlifescienceai/medmcqa",
      image: "/assets/examples/medmcqa-${colorscheme}.jpg",
      data: "example://medmcqa",
      settings: {
        text: "question",
        embedding: { precomputed: { x: "projection_x", y: "projection_y", neighbors: "neighbors" } },
      },
      state: {
        version: "0.21.0",
        charts: {
          1: {
            type: "embedding",
            title: "Embedding",
            data: { x: "projection_x", y: "projection_y", text: "question", category: "subject_name" },
          },
          2: { title: "topic_name", type: "count-plot", data: { field: "topic_name" } },
          3: { type: "instances", title: "Table" },
          4: { type: "count-plot", title: "subject_name", data: { field: "subject_name" } },
        },
        currentLayout: "1",
        layouts: {
          1: {
            type: "dashboard",
            name: "Dashboard",
            chartIds: ["1", "2", "3", "4"],
            numRows: 16,
            numColumns: 24,
            grids: {
              "24x16": {
                placements: {
                  1: { x: 0, y: 0, width: 12, height: 16 },
                  2: { x: 18, y: 0, width: 6, height: 5 },
                  3: { x: 12, y: 5, width: 12, height: 11 },
                  4: { x: 12, y: 0, width: 6, height: 5 },
                },
              },
            },
          },
        },
      },
    },
    {
      title: "SuperGPQA",
      details: "Data from Hugging Face: m-a-p/SuperGPQA",
      image: "/assets/examples/supergpqa-${colorscheme}.jpg",
      data: "example://SuperGPQA",
      settings: {
        text: "question",
        embedding: { precomputed: { x: "projection_x", y: "projection_y", neighbors: "neighbors" } },
      },
      state: {
        version: "0.21.0",
        charts: {
          1: {
            type: "embedding",
            title: "Embedding",
            data: { x: "projection_x", y: "projection_y", text: "question", category: "discipline" },
          },
          2: {
            cardTemplate:
              '<div class="qa-card" style="padding: 14px 16px; display: flex; flex-direction: column; gap: 10px; color: var(--qa-text); background: var(--qa-bg); font-family: -apple-system, BlinkMacSystemFont, \'Segoe UI\', sans-serif; font-size: 13px; line-height: 1.5;">\n  <style>\n    .qa-card {\n      --qa-bg: transparent;\n      --qa-text: #0f172a;\n      --qa-muted: #64748b;\n      --qa-disc-bg: #eef2ff; --qa-disc-text: #4338ca;\n      --qa-field-bg: #ecfeff; --qa-field-text: #0e7490;\n      --qa-sub-bg: #faf5ff; --qa-sub-text: #7e22ce;\n      --qa-easy-bg: #dcfce7; --qa-easy-text: #166534;\n      --qa-mid-bg: #fef9c3; --qa-mid-text: #854d0e;\n      --qa-hard-bg: #fee2e2; --qa-hard-text: #991b1b;\n      --qa-calc-bg: #fff7ed; --qa-calc-text: #9a3412;\n      --qa-opt-bg: #f8fafc; --qa-opt-border: #e2e8f0; --qa-opt-text: #334155;\n      --qa-letter-bg: #e2e8f0; --qa-letter-text: #475569;\n      --qa-correct-bg: #dcfce7; --qa-correct-border: #16a34a; --qa-correct-text: #14532d;\n      --qa-correct-letter-bg: #16a34a; --qa-correct-letter-text: #ffffff;\n    }\n    .qa-card:is(.dark *) {\n      --qa-text: #f1f5f9;\n      --qa-muted: #94a3b8;\n      --qa-disc-bg: #312e81; --qa-disc-text: #c7d2fe;\n      --qa-field-bg: #083344; --qa-field-text: #67e8f9;\n      --qa-sub-bg: #3b0764; --qa-sub-text: #d8b4fe;\n      --qa-easy-bg: #14532d; --qa-easy-text: #86efac;\n      --qa-mid-bg: #422006; --qa-mid-text: #fde68a;\n      --qa-hard-bg: #450a0a; --qa-hard-text: #fca5a5;\n      --qa-calc-bg: #431407; --qa-calc-text: #fed7aa;\n      --qa-opt-bg: #1e293b; --qa-opt-border: #334155; --qa-opt-text: #cbd5e1;\n      --qa-letter-bg: #334155; --qa-letter-text: #cbd5e1;\n      --qa-correct-bg: #052e16; --qa-correct-border: #22c55e; --qa-correct-text: #bbf7d0;\n      --qa-correct-letter-bg: #22c55e; --qa-correct-letter-text: #052e16;\n    }\n    .qa-card .qa-badge { display: inline-flex; align-items: center; gap: 4px; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 600; line-height: 1.4; white-space: nowrap; }\n    .qa-card .qa-meta { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }\n    .qa-card .qa-question { font-size: 14px; font-weight: 400; color: var(--qa-text); }\n    .qa-card .qa-options { display: flex; flex-direction: column; gap: 3px; }\n    .qa-card .qa-option { display: flex; align-items: center; gap: 6px; padding: 2px 8px; background: var(--qa-opt-bg); border: 1px solid var(--qa-opt-border); border-radius: 6px; color: var(--qa-opt-text); line-height: 1.35; }\n    .qa-card .qa-option.qa-correct { background: var(--qa-correct-bg); border-color: var(--qa-correct-border); color: var(--qa-correct-text); font-weight: 500; }\n    .qa-card .qa-letter { flex: 0 0 auto; width: 16px; height: 16px; display: inline-flex; align-items: center; justify-content: center; border-radius: 4px; background: var(--qa-letter-bg); color: var(--qa-letter-text); font-weight: 700; font-size: 10px; }\n    .qa-card .qa-option.qa-correct .qa-letter { background: var(--qa-correct-letter-bg); color: var(--qa-correct-letter-text); }\n    .qa-card .qa-text { flex: 1; font-size: 12px; }\n    .qa-card .qa-foot { display: flex; justify-content: space-between; align-items: center; gap: 8px; padding-top: 4px; border-top: 1px dashed var(--qa-opt-border); font-size: 11px; color: var(--qa-muted); }\n    .qa-card .qa-uuid { font-family: ui-monospace, SFMono-Regular, monospace; }\n  </style>\n  {% assign letters = "A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z" | split: "," %}\n  <div class="qa-meta">\n    <span class="qa-badge" style="background: var(--qa-disc-bg); color: var(--qa-disc-text);">{{ discipline }}</span>\n    <span class="qa-badge" style="background: var(--qa-field-bg); color: var(--qa-field-text);">{{ field }}</span>\n    <span class="qa-badge" style="background: var(--qa-sub-bg); color: var(--qa-sub-text);">{{ subfield }}</span>\n    {% if difficulty == "easy" %}<span class="qa-badge" style="background: var(--qa-easy-bg); color: var(--qa-easy-text);">● easy</span>{% elsif difficulty == "middle" %}<span class="qa-badge" style="background: var(--qa-mid-bg); color: var(--qa-mid-text);">● middle</span>{% elsif difficulty == "hard" %}<span class="qa-badge" style="background: var(--qa-hard-bg); color: var(--qa-hard-text);">● hard</span>{% endif %}\n    {% if is_calculation %}<span class="qa-badge" style="background: var(--qa-calc-bg); color: var(--qa-calc-text);">∑ calculation</span>{% endif %}\n  </div>\n  <div class="qa-question">{{ question }}</div>\n  <div class="qa-options">\n    {% for opt in options %}{% assign letter = letters[forloop.index0] %}<div class="qa-option{% if letter == answer_letter %} qa-correct{% endif %}"><span class="qa-letter">{{ letter }}</span><span class="qa-text">{{ opt }}</span></div>{% endfor %}\n  </div>\n  <div class="qa-foot">\n    <span>Correct answer: <strong style="color: var(--qa-correct-border);">{{ answer_letter }}</strong></span>\n    <span class="qa-uuid">{{ uuid | slice: 0, 8 }}</span>\n  </div>\n</div>',
            columns: [
              "uuid",
              "question",
              "options",
              "answer",
              "answer_letter",
              "discipline",
              "field",
              "subfield",
              "difficulty",
              "is_calculation",
            ],
            pageSize: 20,
            title: "Questions",
            type: "instances",
            viewMode: "cards",
          },
          3: {
            type: "count-plot",
            title: "discipline",
            data: {
              field: "discipline",
            },
          },
          4: {
            type: "count-plot",
            title: "field",
            data: {
              field: "field",
            },
          },
          5: {
            type: "count-plot",
            title: "subfield",
            data: {
              field: "subfield",
            },
          },
          6: {
            type: "count-plot",
            title: "difficulty",
            data: {
              field: "difficulty",
            },
          },
          7: {
            type: "count-plot",
            title: "is_calculation",
            data: {
              field: "is_calculation",
            },
          },
        },
        layouts: {
          1: {
            type: "dashboard",
            name: "SuperGPQA Dashboard",
            chartIds: ["1", "2", "3", "4", "5", "6", "7"],
            numRows: 16,
            numColumns: 24,
            grids: {
              "24x16": {
                placements: {
                  1: { x: 0, y: 0, width: 12, height: 16 },
                  2: { x: 16, y: 0, width: 8, height: 16 },
                  3: { x: 12, y: 0, width: 4, height: 4 },
                  4: { x: 12, y: 4, width: 4, height: 4 },
                  5: { x: 12, y: 8, width: 4, height: 4 },
                  6: { x: 12, y: 12, width: 4, height: 2 },
                  7: { x: 12, y: 14, width: 4, height: 2 },
                },
              },
            },
          },
        },
        currentLayout: "1",
      },
    },
    {
      title: "ImageWoof",
      details: "Data from Hugging Face: frgfm/imagewoof",
      image: "/assets/examples/imagewoof-${colorscheme}.jpg",
      data: "example://imagewoof",
      settings: {
        embedding: { precomputed: { x: "projection_x", y: "projection_y", neighbors: "neighbors" } },
      },
      state: {
        version: "0.21.0",
        charts: {
          "1": {
            type: "embedding",
            title: "Embedding",
            data: { x: "projection_x", y: "projection_y", neighbors: "neighbors", category: "label" },
          },
          "2": { type: "instances", title: "Instances", viewMode: "cards" },
        },
        layouts: {
          "1": {
            type: "dashboard",
            name: "Dashboard",
            chartIds: ["1", "2"],
            grids: {
              "24x18": {
                placements: {
                  "1": { x: 0, y: 0, width: 13, height: 18 },
                  "2": { x: 13, y: 0, width: 11, height: 18 },
                },
              },
            },
          },
        },
        columnStyles: {
          image: { display: "full", renderer: "image", options: { size: 160 } },
          neighbors: { display: "hidden" },
          projection_y: { display: "hidden" },
          projection_x: { display: "hidden" },
        },
      },
    },
  ],
  tabular: [
    {
      title: "Movies Dashboard",
      details: "Data from vega-datasets",
      image: "/assets/examples/movies-dashboard-${colorscheme}.jpg",
      data: "example://movies",
      settings: {},
      state: {
        version: "0.21.0",
        charts: {
          1: {
            title: "IMDB Rating vs. Rotten Tomatos Rating",
            layers: [
              {
                mark: "rect",
                filter: "$filter",
                width: 1,
                style: { fillColor: "$ruleColor" },
                encoding: {
                  x: { field: "IMDB Rating" },
                  y1: { aggregate: "min", field: "Rotten Tomatoes Rating" },
                  y2: { aggregate: "max", field: "Rotten Tomatoes Rating" },
                },
              },
              {
                mark: "rect",
                filter: "$filter",
                width: { gap: 1, clampToRatio: 0.1 },
                encoding: {
                  x: { field: "IMDB Rating" },
                  y1: { aggregate: "quantile", quantile: 0.25, field: "Rotten Tomatoes Rating" },
                  y2: { aggregate: "quantile", quantile: 0.75, field: "Rotten Tomatoes Rating" },
                },
              },
              {
                mark: "rect",
                filter: "$filter",
                height: 1,
                width: { gap: 1, clampToRatio: 0.1 },
                style: { fillColor: "$ruleColor" },
                encoding: { x: { field: "IMDB Rating" }, y: { aggregate: "median", field: "Rotten Tomatoes Rating" } },
              },
            ],
            selection: { brush: { encoding: "x" } },
            axis: { y: { title: "Rotten Tomatoes Rating" } },
            widgets: [
              { type: "scale.type", channel: "x" },
              { type: "scale.type", channel: "y" },
            ],
          },
          2: { type: "instances", title: "Table" },
          3: {
            title: "US Gross",
            layers: [
              {
                mark: "bar",
                style: { fillColor: "$markColorFade" },
                encoding: { x: { field: "US Gross" }, y: { aggregate: "count" } },
              },
              { mark: "bar", filter: "$filter", encoding: { x: { field: "US Gross" }, y: { aggregate: "count" } } },
            ],
            selection: { brush: { encoding: "x" } },
            widgets: [
              { type: "scale.type", channel: "x" },
              { type: "encoding.normalize", attribute: "y", layer: [0, 1], options: ["x"] },
            ],
          },
          4: {
            title: "Worldwide Gross",
            layers: [
              {
                mark: "bar",
                style: { fillColor: "$markColorFade" },
                encoding: { x: { field: "Worldwide Gross" }, y: { aggregate: "count" } },
              },
              {
                mark: "bar",
                filter: "$filter",
                encoding: { x: { field: "Worldwide Gross" }, y: { aggregate: "count" } },
              },
            ],
            selection: { brush: { encoding: "x" } },
            widgets: [
              { type: "scale.type", channel: "x" },
              { type: "encoding.normalize", attribute: "y", layer: [0, 1], options: ["x"] },
            ],
          },
          5: {
            title: "Production Budget by Rotten Tomatoes Rating",
            layers: [
              {
                mark: "rect",
                filter: "$filter",
                width: 1,
                style: { fillColor: "$ruleColor" },
                encoding: {
                  x: { field: "Production Budget" },
                  y1: { aggregate: "min", field: "Rotten Tomatoes Rating" },
                  y2: { aggregate: "max", field: "Rotten Tomatoes Rating" },
                },
              },
              {
                mark: "rect",
                filter: "$filter",
                width: { gap: 1, clampToRatio: 0.1 },
                encoding: {
                  x: { field: "Production Budget" },
                  y1: { aggregate: "quantile", quantile: 0.25, field: "Rotten Tomatoes Rating" },
                  y2: { aggregate: "quantile", quantile: 0.75, field: "Rotten Tomatoes Rating" },
                },
              },
              {
                mark: "rect",
                filter: "$filter",
                height: 1,
                width: { gap: 1, clampToRatio: 0.1 },
                style: { fillColor: "$ruleColor" },
                encoding: {
                  x: { field: "Production Budget" },
                  y: { aggregate: "median", field: "Rotten Tomatoes Rating" },
                },
              },
            ],
            selection: { brush: { encoding: "x" } },
            axis: { y: { title: "Rotten Tomatoes Rating" } },
            widgets: [
              { type: "scale.type", channel: "x" },
              { type: "scale.type", channel: "y" },
            ],
          },
          6: {
            title: "Production Budget",
            layers: [
              {
                mark: "bar",
                style: { fillColor: "$markColorFade" },
                encoding: { x: { field: "Production Budget" }, y: { aggregate: "count" } },
              },
              {
                mark: "bar",
                filter: "$filter",
                encoding: { x: { field: "Production Budget" }, y: { aggregate: "count" } },
              },
            ],
            selection: { brush: { encoding: "x" } },
            widgets: [
              { type: "scale.type", channel: "x" },
              { type: "encoding.normalize", attribute: "y", layer: [0, 1], options: ["x"] },
            ],
          },
          7: {
            title: "Production Budget",
            layers: [
              {
                mark: "rect",
                filter: "$filter",
                width: 1,
                style: { fillColor: "$ruleColor" },
                encoding: {
                  x: { field: "Production Budget" },
                  y1: { aggregate: "min", field: "US Gross" },
                  y2: { aggregate: "max", field: "US Gross" },
                },
              },
              {
                mark: "rect",
                filter: "$filter",
                width: { gap: 1, clampToRatio: 0.1 },
                encoding: {
                  x: { field: "Production Budget" },
                  y1: { aggregate: "quantile", quantile: 0.25, field: "US Gross" },
                  y2: { aggregate: "quantile", quantile: 0.75, field: "US Gross" },
                },
              },
              {
                mark: "rect",
                filter: "$filter",
                height: 1,
                width: { gap: 1, clampToRatio: 0.1 },
                style: { fillColor: "$ruleColor" },
                encoding: { x: { field: "Production Budget" }, y: { aggregate: "median", field: "US Gross" } },
              },
            ],
            selection: { brush: { encoding: "x" } },
            axis: { y: { title: "US Gross" } },
            widgets: [
              { type: "scale.type", channel: "x" },
              { type: "scale.type", channel: "y" },
            ],
            scale: { x: { type: "linear" }, y: { type: "linear" } },
          },
          8: { type: "count-plot", title: "Major Genre", data: { field: "Major Genre" }, limit: 100, labels: "%" },
          9: {
            title: "IMDB Rating",
            layers: [
              {
                mark: "bar",
                style: { fillColor: "$markColorFade" },
                encoding: { x: { field: "IMDB Rating" }, y: { aggregate: "count" } },
              },
              { mark: "bar", filter: "$filter", encoding: { x: { field: "IMDB Rating" }, y: { aggregate: "count" } } },
            ],
            selection: { brush: { encoding: "x" } },
            widgets: [
              { type: "scale.type", channel: "x" },
              { type: "encoding.normalize", attribute: "y", layer: [0, 1], options: ["x"] },
            ],
          },
        },
        currentLayout: "1",
        layouts: {
          1: {
            type: "dashboard",
            name: "Dashboard",
            chartIds: ["1", "2", "3", "4", "5", "6", "7", "8", "9"],
            numRows: 16,
            numColumns: 24,
            grids: {
              "24x16": {
                placements: {
                  1: { x: 10, y: 0, width: 5, height: 5 },
                  2: { x: 11, y: 5, width: 13, height: 11 },
                  3: { x: 0, y: 0, width: 5, height: 5 },
                  4: { x: 0, y: 5, width: 5, height: 5 },
                  5: { x: 15, y: 0, width: 9, height: 5 },
                  6: { x: 5, y: 0, width: 5, height: 5 },
                  7: { x: 5, y: 10, width: 6, height: 6 },
                  8: { x: 0, y: 10, width: 5, height: 6 },
                  9: { x: 5, y: 5, width: 6, height: 5 },
                },
              },
            },
          },
        },
      },
    },
    {
      title: "ScienceQA",
      details: "Data from Hugging Face: derek-thomas/ScienceQA",
      image: "/assets/examples/scienceqa-${colorscheme}.jpg",
      data: "example://ScienceQA",
      settings: {},
      state: {
        version: "0.21.0",
        charts: {
          1: { type: "predicates", title: "SQL Predicates" },
          2: { type: "instances", title: "Table" },
          3: {
            title: "grade, topic",
            plotSize: { height: 350 },
            layers: [
              {
                mark: "rect",
                filter: "$filter",
                zIndex: -1,
                encoding: {
                  x: { field: "grade" },
                  y: { field: "topic", bin: { desiredCount: 100 } },
                  color: { aggregate: "count", normalize: "y" },
                },
              },
              { mark: "rect", zIndex: -2, encoding: { color: { value: 0 } } },
            ],
            selection: { brush: { encoding: "xy" } },
            scale: {
              x: {
                domain: [
                  "grade1",
                  "grade2",
                  "grade3",
                  "grade4",
                  "grade5",
                  "grade6",
                  "grade7",
                  "grade8",
                  "grade9",
                  "grade10",
                  "grade11",
                  "grade12",
                ],
              },
            },
            widgets: [
              { type: "scale.type", channel: "x" },
              { type: "scale.type", channel: "y" },
              { type: "encoding.normalize", attribute: "color", layer: 0, options: ["x", "y"] },
            ],
          },
          4: { type: "count-plot", title: "task", data: { field: "task" } },
          5: { type: "count-plot", title: "grade", data: { field: "grade" } },
          6: { type: "count-plot", title: "subject", data: { field: "subject" } },
          7: { type: "count-plot", title: "topic", data: { field: "topic" } },
          8: { type: "count-plot", title: "category", data: { field: "category" } },
          9: { type: "count-plot", title: "skill", data: { field: "skill" }, limit: 10 },
          10: { type: "count-plot", title: "lecture", data: { field: "lecture" }, limit: 10 },
        },
      },
    },
    {
      title: "Census Income",
      details: "Data from Hugging Face: scikit-learn/adult-census-income",
      image: "/assets/examples/census-income-${colorscheme}.jpg",
      data: "example://census-income",
      settings: {},
      state: {
        version: "0.21.0",
        charts: {
          1: {
            title: "Age by Marital Status",
            layers: [
              {
                mark: "bar",
                style: { fillColor: "$markColorFade" },
                encoding: { x: { field: "age" }, y: { aggregate: "count" } },
              },
              {
                mark: "bar",
                filter: "$filter",
                encoding: { x: { field: "age" }, y: { aggregate: "count" }, color: { field: "marital.status" } },
              },
            ],
            selection: { brush: { encoding: "x" } },
            widgets: [
              { type: "scale.type", channel: "x" },
              { type: "encoding.normalize", attribute: "y", layer: [0, 1], options: ["x"] },
            ],
          },
          2: { type: "instances", title: "Table" },
          3: { title: "Sex", type: "count-plot", data: { field: "sex" } },
          4: { type: "count-plot", title: "Workclass", data: { field: "workclass" } },
          5: {
            title: "Age (CDF) by Income",
            layers: [
              {
                mark: "line",
                filter: "$filter",
                encoding: {
                  x: { aggregate: "ecdf-value", field: "age" },
                  y: { aggregate: "ecdf-rank" },
                  color: { field: "education" },
                },
              },
            ],
            selection: { brush: { encoding: "x" } },
            widgets: [{ type: "scale.type", channel: "x" }],
          },
          6: {
            title: "Age (CDF) by Martial Status",
            layers: [
              {
                mark: "line",
                filter: "$filter",
                encoding: {
                  x: { aggregate: "ecdf-value", field: "age" },
                  y: { aggregate: "ecdf-rank" },
                  color: { field: "marital.status" },
                },
              },
            ],
            selection: { brush: { encoding: "x" } },
            widgets: [{ type: "scale.type", channel: "x" }],
          },
          7: {
            title: "Age by Income",
            layers: [
              {
                mark: "bar",
                style: { fillColor: "$markColorFade" },
                encoding: { x: { field: "age" }, y: { aggregate: "count" } },
              },
              {
                mark: "bar",
                filter: "$filter",
                encoding: { x: { field: "age" }, y: { aggregate: "count" }, color: { field: "income" } },
              },
            ],
            selection: { brush: { encoding: "x" } },
            widgets: [
              { type: "scale.type", channel: "x" },
              { type: "encoding.normalize", attribute: "y", layer: [0, 1], options: ["x"] },
            ],
          },
        },
        currentLayout: "1",
        layouts: {
          1: {
            type: "dashboard",
            name: "Dashboard",
            chartIds: ["1", "2", "3", "4", "5", "6", "7"],
            numRows: 16,
            numColumns: 24,
            grids: {
              "24x16": {
                placements: {
                  1: { x: 0, y: 0, width: 7, height: 6 },
                  2: { x: 14, y: 0, width: 10, height: 16 },
                  3: { x: 7, y: 12, width: 7, height: 4 },
                  4: { x: 0, y: 12, width: 7, height: 4 },
                  5: { x: 7, y: 6, width: 7, height: 6 },
                  6: { x: 0, y: 6, width: 7, height: 6 },
                  7: { x: 7, y: 0, width: 7, height: 6 },
                },
              },
            },
          },
        },
      },
    },
    {
      title: "California Housing",
      details: "Data from Hugging Face: gvlassis/california_housing",
      image: "/assets/examples/california-housing-${colorscheme}.jpg",
      data: "example://california-housing",
      settings: {},
      state: {
        version: "0.21.0",
        charts: {
          1: {
            title: "MedHouseVal",
            layers: [
              {
                mark: "bar",
                style: { fillColor: "$markColorFade" },
                encoding: { x: { field: "MedHouseVal" }, y: { aggregate: "count" } },
              },
              {
                mark: "bar",
                filter: "$filter",
                encoding: { x: { field: "MedHouseVal" }, y: { aggregate: "count" }, color: { field: "HouseAge" } },
              },
            ],
            selection: { brush: { encoding: "x" } },
            widgets: [
              { type: "scale.type", channel: "x" },
              { type: "encoding.normalize", attribute: "y", layer: [0, 1], options: ["x"] },
            ],
          },
          2: { type: "instances", title: "Table" },
          3: {
            title: "MedInc",
            layers: [
              {
                mark: "bar",
                style: { fillColor: "$markColorFade" },
                encoding: { x: { field: "MedInc" }, y: { aggregate: "count" } },
              },
              { mark: "bar", filter: "$filter", encoding: { x: { field: "MedInc" }, y: { aggregate: "count" } } },
            ],
            selection: { brush: { encoding: "x" } },
            widgets: [
              { type: "scale.type", channel: "x" },
              { type: "encoding.normalize", attribute: "y", layer: [0, 1], options: ["x"] },
            ],
          },
          4: {
            title: "HouseAge",
            layers: [
              {
                mark: "bar",
                style: { fillColor: "$markColorFade" },
                encoding: { x: { field: "HouseAge" }, y: { aggregate: "count" } },
              },
              { mark: "bar", filter: "$filter", encoding: { x: { field: "HouseAge" }, y: { aggregate: "count" } } },
            ],
            selection: { brush: { encoding: "x" } },
            widgets: [
              { type: "scale.type", channel: "x" },
              { type: "encoding.normalize", attribute: "y", layer: [0, 1], options: ["x"] },
            ],
          },
          5: {
            title: "Population",
            layers: [
              {
                mark: "bar",
                style: { fillColor: "$markColorFade" },
                encoding: { x: { field: "Population" }, y: { aggregate: "count" } },
              },
              { mark: "bar", filter: "$filter", encoding: { x: { field: "Population" }, y: { aggregate: "count" } } },
            ],
            selection: { brush: { encoding: "x" } },
            widgets: [
              { type: "scale.type", channel: "x" },
              { type: "encoding.normalize", attribute: "y", layer: [0, 1], options: ["x"] },
            ],
          },
          6: {
            title: "MedHouseVal",
            layers: [
              {
                mark: "bar",
                style: { fillColor: "$markColorFade" },
                encoding: { x: { field: "MedHouseVal" }, y: { aggregate: "count" } },
              },
              { mark: "bar", filter: "$filter", encoding: { x: { field: "MedHouseVal" }, y: { aggregate: "count" } } },
            ],
            selection: { brush: { encoding: "x" } },
            widgets: [
              { type: "scale.type", channel: "x" },
              { type: "encoding.normalize", attribute: "y", layer: [0, 1], options: ["x"] },
            ],
          },
          7: {
            type: "embedding",
            title: "Embedding",
            data: { x: "Longitude", y: "Latitude", category: "MedHouseVal" },
            mode: "points",
          },
        },
        currentLayout: "1",
        layouts: {
          1: {
            type: "dashboard",
            name: "Dashboard",
            chartIds: ["1", "2", "3", "4", "5", "6", "7"],
            numRows: 16,
            numColumns: 24,
            grids: {
              "24x16": {
                placements: {
                  1: { x: 17, y: 0, width: 7, height: 10 },
                  2: { x: 0, y: 10, width: 24, height: 6 },
                  3: { x: 7, y: 5, width: 5, height: 5 },
                  4: { x: 12, y: 0, width: 5, height: 5 },
                  5: { x: 7, y: 0, width: 5, height: 5 },
                  6: { x: 12, y: 5, width: 5, height: 5 },
                  7: { x: 0, y: 0, width: 7, height: 10 },
                },
              },
            },
          },
        },
      },
    },
  ],
};

const datasets = [
  {
    key: "example://imagewoof",
    title: "Imagewoof: a subset of 10 dog breed classes from ImageNet",
    authors: "Jeremy Howard, 2019",
    link: {
      title: "frgfm/imagewoof",
      url: "https://huggingface.co/datasets/frgfm/imagewoof",
    },
  },
  {
    key: "example://wine-reviews",
    title: "Wine Reviews",
    authors: "Zackthoutt, spawn99",
    link: {
      title: "spawn99/wine-reviews",
      url: "https://huggingface.co/datasets/spawn99/wine-reviews",
    },
  },
  {
    key: "example://medmcqa",
    title: "MedMCQA: A Large-scale Multi-Subject Multi-Choice Dataset for Medical domain Question Answering",
    authors: "Ankit Pal, Logesh Kumar Umapathi, Malaikannan Sankarasubbu, 2022",
    link: {
      title: "openlifescienceai/medmcqa",
      url: "https://huggingface.co/datasets/openlifescienceai/medmcqa",
    },
  },
  {
    key: "example://SuperGPQA",
    title: "SuperGPQA: Scaling LLM Evaluation across 285 Graduate Disciplines",
    authors: "M-A-P Team et al., 2025",
    link: {
      title: "m-a-p/SuperGPQA",
      url: "https://huggingface.co/datasets/m-a-p/SuperGPQA",
    },
  },
  {
    key: "example://vispubdata",
    title: "vispubdata.org: A Metadata Collection about IEEE Visualization (VIS) Publications",
    authors: "Petra Isenberg et al., 2017",
    link: {
      title: "vispubdata.org",
      url: "https://www.vispubdata.org/",
    },
  },
  {
    key: "example://census-income",
    title: "Adult Census Income",
    authors: "UCI machine learning repository",
    link: {
      title: "scikit-learn/adult-census-income",
      url: "https://huggingface.co/datasets/scikit-learn/adult-census-income",
    },
  },
  {
    key: "example://california-housing",
    title: "California Housing",
    authors: 'The California Housing dataset, first appearing in "Sparse spatial autoregressions" (1997)',
    link: {
      title: "gvlassis/california_housing",
      url: "https://huggingface.co/datasets/gvlassis/california_housing",
    },
  },
  {
    key: "example://ScienceQA",
    title: "Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering",
    authors:
      "Pan Lu, Swaroop Mishra, Tony Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, Ashwin Kalyan, 2022",
    link: {
      title: "derek-thomas/ScienceQA",
      url: "https://huggingface.co/datasets/derek-thomas/ScienceQA",
    },
  },
  {
    key: "example://movies",
    title: "Movies",
    authors: "vega-datasets",
    link: {
      title: "vega-datasets/movies",
      url: "https://github.com/vega/vega-datasets/blob/main/datapackage.md#movies",
    },
  },
];

async function process(example: Example) {
  return {
    title: example.title,
    details: example.details,
    image: {
      light: example.image.replace("${colorscheme}", "light"),
      dark: example.image.replace("${colorscheme}", "dark"),
    },
    data: example.data,
    settings:
      example.settings == null
        ? undefined
        : typeof example.settings == "string"
          ? example.settings
          : await encode(example.settings),
    state:
      example.state == null
        ? undefined
        : typeof example.state == "string"
          ? example.state
          : await encode(example.state),
  };
}

export default {
  async load() {
    let result: any = {};
    for (let key in examples) {
      result[key] = await Promise.all(examples[key].map(process));
    }
    return { examples: result, datasets };
  },
};
