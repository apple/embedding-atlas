import type { BuiltinChartSpec, EmbeddingAtlasState } from "embedding-atlas";

import { encode } from "./utils.js";

interface Example {
  title: string;
  details: string;
  image: string;
  data: string;
  settings?: any;
  state?: Omit<EmbeddingAtlasState, "timestamp" | "charts"> & { charts: Record<string, BuiltinChartSpec> };
}

const examples: Record<string, Example[]> = {
  embedding: [
    {
      title: "Wine Reviews",
      details: "Data from Hugging Face: spawn99/wine-reviews",
      image: "/assets/examples/wine-reviews.jpg",
      data: "example://wine-reviews",
      settings: {
        text: "description",
        embedding: { precomputed: { x: "projection_x", y: "projection_y", neighbors: "neighbors" } },
      },
      state: {
        version: "0.15.0",
        charts: {
          1: { type: "embedding", title: "Embedding", data: { x: "projection_x", y: "projection_y", text: "description", category: "country" } },
          2: { type: "predicates", title: "SQL Predicates" },
          3: {
            title: "points vs. price",
            layers: [
              { mark: "rect", filter: "$filter", width: 1, style: { fillColor: "$ruleColor" }, encoding: { x: { field: "price" }, y1: { aggregate: "min", field: "points" }, y2: { aggregate: "max", field: "points" } } },
              { mark: "rect", filter: "$filter", width: { gap: 1, clampToRatio: 0.1 }, encoding: { x: { field: "price" }, y1: { aggregate: "quantile", quantile: 0.25, field: "points" }, y2: { aggregate: "quantile", quantile: 0.75, field: "points" } } },
              { mark: "rect", filter: "$filter", height: 1, width: { gap: 1, clampToRatio: 0.1 }, style: { fillColor: "$ruleColor" }, encoding: { x: { field: "price" }, y: { aggregate: "median", field: "points" } } },
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
              { mark: "bar", style: { fillColor: "$markColorFade" }, encoding: { x: { field: "price" }, y: { aggregate: "count" } } },
              { mark: "bar", filter: "$filter", encoding: { x: { field: "price" }, y: { aggregate: "count" }, color: { field: "country" } } },
            ],
            selection: { brush: { encoding: "x" } },
            widgets: [
              { type: "scale.type", channel: "x" },
              { type: "encoding.normalize", attribute: "y", layer: [0, 1], options: ["x"] },
            ],
          },
          5: { type: "table", title: "Table", columns: ["title", "country", "province", "description", "points", "price", "variety", "designation", "projection_x", "projection_y", "neighbors"] },
          6: { type: "count-plot", title: "country", data: { field: "country" } },
          7: { type: "count-plot", title: "province", data: { field: "province" } },
          8: {
            title: "points",
            layers: [
              { mark: "bar", style: { fillColor: "$markColorFade" }, encoding: { x: { field: "points" }, y: { aggregate: "count" } } },
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
              { mark: "bar", style: { fillColor: "$markColorFade" }, encoding: { x: { field: "price" }, y: { aggregate: "count" } } },
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
      image: "/assets/examples/vispubdata.jpg",
      data: "example://vispubdata",
      settings: {
        text: "Abstract",
        embedding: { precomputed: { x: "projection_x", y: "projection_y", neighbors: "neighbors" } },
      },
      state: {
        version: "0.15.0",
        charts: {
          1: { type: "embedding", title: "Embedding", data: { x: "projection_x", y: "projection_y", text: "Abstract", category: "Conference" } },
          2: { type: "predicates", title: "SQL Predicates" },
          3: { type: "table", title: "Table", columns: ["Year", "Conference", "Title", "Abstract", "Link", "PaperType", "Award", "AuthorNames_Deduped", "AuthorAffiliation", "AuthorKeywords", "AminerCitationCount", "CitationCount_CrossRef", "PubsCited_CrossRef", "projection_x", "projection_y", "neighbors"] },
          4: {
            title: "Year",
            layers: [
              { mark: "bar", style: { fillColor: "$markColorFade" }, encoding: { x: { field: "Year" }, y: { aggregate: "count" } } },
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
              { mark: "bar", style: { fillColor: "$markColorFade" }, encoding: { x: { field: "AminerCitationCount" }, y: { aggregate: "count" } } },
              { mark: "bar", filter: "$filter", encoding: { x: { field: "AminerCitationCount" }, y: { aggregate: "count" } } },
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
              { mark: "bar", style: { fillColor: "$markColorFade" }, encoding: { x: { field: "CitationCount_CrossRef" }, y: { aggregate: "count" } } },
              { mark: "bar", filter: "$filter", encoding: { x: { field: "CitationCount_CrossRef" }, y: { aggregate: "count" } } },
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
              { mark: "bar", style: { fillColor: "$markColorFade" }, encoding: { x: { field: "PubsCited_CrossRef" }, y: { aggregate: "count" } } },
              { mark: "bar", filter: "$filter", encoding: { x: { field: "PubsCited_CrossRef" }, y: { aggregate: "count" } } },
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
      image: "/assets/examples/medmcqa.jpg",
      data: "example://medmcqa",
      settings: {
        text: "question",
        embedding: { precomputed: { x: "projection_x", y: "projection_y", neighbors: "neighbors" } },
      },
      state: {
        version: "0.15.0",
        charts: {
          1: { type: "embedding", title: "Embedding", data: { x: "projection_x", y: "projection_y", text: "question", category: "subject_name" } },
          2: { title: "topic_name", type: "count-plot", data: { field: "topic_name" } },
          3: { type: "table", title: "Table", columns: ["id", "question", "opa", "opb", "opc", "opd", "cop", "choice_type", "exp", "subject_name", "topic_name", "projection_x", "projection_y", "neighbors"] },
          4: { type: "count-plot", title: "subject_name", data: { field: "subject_name" } },
        },
        layout: "dashboard",
        layoutStates: {
          dashboard: {
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
      details: "Data from Hugging Face: openlifescienceai/medmcqa",
      image: "/assets/examples/supergpqa.jpg",
      data: "example://SuperGPQA",
      settings: {
        text: "question",
        embedding: { precomputed: { x: "projection_x", y: "projection_y", neighbors: "neighbors" } },
      },
      state: {
        version: "0.15.0",
        charts: {
          1: { type: "embedding", title: "Embedding", data: { x: "projection_x", y: "projection_y", text: "question", category: "discipline" } },
          2: { type: "table", title: "Table", columns: ["uuid", "question", "options", "answer", "answer_letter", "discipline", "field", "subfield", "difficulty", "is_calculation", "projection_x", "projection_y", "neighbors"] },
          3: { type: "count-plot", title: "discipline", data: { field: "discipline" } },
          4: { type: "count-plot", title: "field", data: { field: "field" } },
          5: { type: "count-plot", title: "subfield", data: { field: "subfield" } },
          6: { type: "count-plot", title: "difficulty", data: { field: "difficulty" } },
          7: { type: "count-plot", title: "is_calculation", data: { field: "is_calculation" } },
        },
        layout: "dashboard",
        layoutStates: {
          dashboard: {
            grids: {
              "24x16": {
                placements: {
                  1: { x: 0, y: 0, width: 11, height: 12 },
                  2: { x: 0, y: 12, width: 24, height: 4 },
                  3: { x: 19, y: 0, width: 5, height: 7 },
                  4: { x: 11, y: 0, width: 8, height: 6 },
                  5: { x: 11, y: 6, width: 8, height: 6 },
                  6: { x: 19, y: 7, width: 5, height: 3 },
                  7: { x: 19, y: 10, width: 5, height: 2 },
                },
              },
            },
          },
        },
      },
    },
  ],
  tabular: [
    {
      title: "Movies Dashboard",
      details: "Data from vega-datasets",
      image: "/assets/examples/movies-dashboard.jpg",
      data: "example://movies",
      settings: {},
      state: {
        version: "0.15.0",
        charts: {
          1: {
            title: "IMDB Rating vs. Rotten Tomatos Rating",
            layers: [
              { mark: "rect", filter: "$filter", width: 1, style: { fillColor: "$ruleColor" }, encoding: { x: { field: "IMDB Rating" }, y1: { aggregate: "min", field: "Rotten Tomatoes Rating" }, y2: { aggregate: "max", field: "Rotten Tomatoes Rating" } } },
              { mark: "rect", filter: "$filter", width: { gap: 1, clampToRatio: 0.1 }, encoding: { x: { field: "IMDB Rating" }, y1: { aggregate: "quantile", quantile: 0.25, field: "Rotten Tomatoes Rating" }, y2: { aggregate: "quantile", quantile: 0.75, field: "Rotten Tomatoes Rating" } } },
              { mark: "rect", filter: "$filter", height: 1, width: { gap: 1, clampToRatio: 0.1 }, style: { fillColor: "$ruleColor" }, encoding: { x: { field: "IMDB Rating" }, y: { aggregate: "median", field: "Rotten Tomatoes Rating" } } },
            ],
            selection: { brush: { encoding: "x" } },
            axis: { y: { title: "Rotten Tomatoes Rating" } },
            widgets: [
              { type: "scale.type", channel: "x" },
              { type: "scale.type", channel: "y" },
            ],
          },
          2: { type: "table", title: "Table", columns: ["Title", "US Gross", "Worldwide Gross", "US DVD Sales", "Production Budget", "Release Date", "MPAA Rating", "Running Time min", "Distributor", "Source", "Major Genre", "Creative Type", "Director", "Rotten Tomatoes Rating", "IMDB Rating", "IMDB Votes"] },
          3: {
            title: "US Gross",
            layers: [
              { mark: "bar", style: { fillColor: "$markColorFade" }, encoding: { x: { field: "US Gross" }, y: { aggregate: "count" } } },
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
              { mark: "bar", style: { fillColor: "$markColorFade" }, encoding: { x: { field: "Worldwide Gross" }, y: { aggregate: "count" } } },
              { mark: "bar", filter: "$filter", encoding: { x: { field: "Worldwide Gross" }, y: { aggregate: "count" } } },
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
              { mark: "rect", filter: "$filter", width: 1, style: { fillColor: "$ruleColor" }, encoding: { x: { field: "Production Budget" }, y1: { aggregate: "min", field: "Rotten Tomatoes Rating" }, y2: { aggregate: "max", field: "Rotten Tomatoes Rating" } } },
              { mark: "rect", filter: "$filter", width: { gap: 1, clampToRatio: 0.1 }, encoding: { x: { field: "Production Budget" }, y1: { aggregate: "quantile", quantile: 0.25, field: "Rotten Tomatoes Rating" }, y2: { aggregate: "quantile", quantile: 0.75, field: "Rotten Tomatoes Rating" } } },
              { mark: "rect", filter: "$filter", height: 1, width: { gap: 1, clampToRatio: 0.1 }, style: { fillColor: "$ruleColor" }, encoding: { x: { field: "Production Budget" }, y: { aggregate: "median", field: "Rotten Tomatoes Rating" } } },
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
              { mark: "bar", style: { fillColor: "$markColorFade" }, encoding: { x: { field: "Production Budget" }, y: { aggregate: "count" } } },
              { mark: "bar", filter: "$filter", encoding: { x: { field: "Production Budget" }, y: { aggregate: "count" } } },
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
              { mark: "rect", filter: "$filter", width: 1, style: { fillColor: "$ruleColor" }, encoding: { x: { field: "Production Budget" }, y1: { aggregate: "min", field: "US Gross" }, y2: { aggregate: "max", field: "US Gross" } } },
              { mark: "rect", filter: "$filter", width: { gap: 1, clampToRatio: 0.1 }, encoding: { x: { field: "Production Budget" }, y1: { aggregate: "quantile", quantile: 0.25, field: "US Gross" }, y2: { aggregate: "quantile", quantile: 0.75, field: "US Gross" } } },
              { mark: "rect", filter: "$filter", height: 1, width: { gap: 1, clampToRatio: 0.1 }, style: { fillColor: "$ruleColor" }, encoding: { x: { field: "Production Budget" }, y: { aggregate: "median", field: "US Gross" } } },
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
              { mark: "bar", style: { fillColor: "$markColorFade" }, encoding: { x: { field: "IMDB Rating" }, y: { aggregate: "count" } } },
              { mark: "bar", filter: "$filter", encoding: { x: { field: "IMDB Rating" }, y: { aggregate: "count" } } },
            ],
            selection: { brush: { encoding: "x" } },
            widgets: [
              { type: "scale.type", channel: "x" },
              { type: "encoding.normalize", attribute: "y", layer: [0, 1], options: ["x"] },
            ],
          },
        },
        layout: "dashboard",
        layoutStates: {
          dashboard: {
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
      image: "/assets/examples/scienceqa.jpg",
      data: "example://ScienceQA",
      settings: {},
      state: {
        version: "0.15.0",
        charts: {
          1: { type: "predicates", title: "SQL Predicates" },
          2: { type: "table", title: "Table", columns: ["image", "question", "choices", "answer", "hint", "task", "grade", "subject", "topic", "category", "skill", "lecture", "solution"] },
          3: {
            title: "grade, topic",
            plotSize: { height: 350 },
            layers: [
              { mark: "rect", filter: "$filter", zIndex: -1, encoding: { x: { field: "grade" }, y: { field: "topic", bin: { desiredCount: 100 } }, color: { aggregate: "count", normalize: "y" } } },
              { mark: "rect", zIndex: -2, encoding: { color: { value: 0 } } },
            ],
            selection: { brush: { encoding: "xy" } },
            scale: { x: { domain: ["grade1", "grade2", "grade3", "grade4", "grade5", "grade6", "grade7", "grade8", "grade9", "grade10", "grade11", "grade12"] } },
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
      image: "/assets/examples/census-income.jpg",
      data: "example://census-income",
      settings: {},
      state: {
        version: "0.15.0",
        charts: {
          1: {
            title: "Age by Marital Status",
            layers: [
              { mark: "bar", style: { fillColor: "$markColorFade" }, encoding: { x: { field: "age" }, y: { aggregate: "count" } } },
              { mark: "bar", filter: "$filter", encoding: { x: { field: "age" }, y: { aggregate: "count" }, color: { field: "marital.status" } } },
            ],
            selection: { brush: { encoding: "x" } },
            widgets: [
              { type: "scale.type", channel: "x" },
              { type: "encoding.normalize", attribute: "y", layer: [0, 1], options: ["x"] },
            ],
          },
          2: { type: "table", title: "Table", columns: ["age", "workclass", "fnlwgt", "education", "education.num", "marital.status", "occupation", "relationship", "race", "sex", "capital.gain", "capital.loss", "hours.per.week", "native.country", "income"] },
          3: { title: "Sex", type: "count-plot", data: { field: "sex" } },
          4: { type: "count-plot", title: "Workclass", data: { field: "workclass" } },
          5: { title: "Age (CDF) by Income", layers: [{ mark: "line", filter: "$filter", encoding: { x: { aggregate: "ecdf-value", field: "age" }, y: { aggregate: "ecdf-rank" }, color: { field: "education" } } }], selection: { brush: { encoding: "x" } }, widgets: [{ type: "scale.type", channel: "x" }] },
          6: {
            title: "Age (CDF) by Martial Status",
            layers: [{ mark: "line", filter: "$filter", encoding: { x: { aggregate: "ecdf-value", field: "age" }, y: { aggregate: "ecdf-rank" }, color: { field: "marital.status" } } }],
            selection: { brush: { encoding: "x" } },
            widgets: [{ type: "scale.type", channel: "x" }],
          },
          7: {
            title: "Age by Income",
            layers: [
              { mark: "bar", style: { fillColor: "$markColorFade" }, encoding: { x: { field: "age" }, y: { aggregate: "count" } } },
              { mark: "bar", filter: "$filter", encoding: { x: { field: "age" }, y: { aggregate: "count" }, color: { field: "income" } } },
            ],
            selection: { brush: { encoding: "x" } },
            widgets: [
              { type: "scale.type", channel: "x" },
              { type: "encoding.normalize", attribute: "y", layer: [0, 1], options: ["x"] },
            ],
          },
        },
        layout: "dashboard",
        layoutStates: {
          dashboard: {
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
      image: "/assets/examples/california-housing.jpg",
      data: "example://california-housing",
      settings: {},
      state: {
        version: "0.15.0",
        charts: {
          1: {
            title: "MedHouseVal",
            layers: [
              { mark: "bar", style: { fillColor: "$markColorFade" }, encoding: { x: { field: "MedHouseVal" }, y: { aggregate: "count" } } },
              { mark: "bar", filter: "$filter", encoding: { x: { field: "MedHouseVal" }, y: { aggregate: "count" }, color: { field: "HouseAge" } } },
            ],
            selection: { brush: { encoding: "x" } },
            widgets: [
              { type: "scale.type", channel: "x" },
              { type: "encoding.normalize", attribute: "y", layer: [0, 1], options: ["x"] },
            ],
          },
          2: { type: "table", title: "Table", columns: ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude", "MedHouseVal"] },
          3: {
            title: "MedInc",
            layers: [
              { mark: "bar", style: { fillColor: "$markColorFade" }, encoding: { x: { field: "MedInc" }, y: { aggregate: "count" } } },
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
              { mark: "bar", style: { fillColor: "$markColorFade" }, encoding: { x: { field: "HouseAge" }, y: { aggregate: "count" } } },
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
              { mark: "bar", style: { fillColor: "$markColorFade" }, encoding: { x: { field: "Population" }, y: { aggregate: "count" } } },
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
              { mark: "bar", style: { fillColor: "$markColorFade" }, encoding: { x: { field: "MedHouseVal" }, y: { aggregate: "count" } } },
              { mark: "bar", filter: "$filter", encoding: { x: { field: "MedHouseVal" }, y: { aggregate: "count" } } },
            ],
            selection: { brush: { encoding: "x" } },
            widgets: [
              { type: "scale.type", channel: "x" },
              { type: "encoding.normalize", attribute: "y", layer: [0, 1], options: ["x"] },
            ],
          },
          7: { type: "embedding", title: "Embedding", data: { x: "Longitude", y: "Latitude", category: "MedHouseVal" }, mode: "points" },
        },
        layout: "dashboard",
        layoutStates: {
          dashboard: {
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
    authors: "Pan Lu, Swaroop Mishra, Tony Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, Ashwin Kalyan, 2022",
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
    image: example.image,
    data: example.data,
    settings: example.settings == null ? undefined : typeof example.settings == "string" ? example.settings : await encode(example.settings),
    state: example.state == null ? undefined : typeof example.state == "string" ? example.state : await encode(example.state),
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
