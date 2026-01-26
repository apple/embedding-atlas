const datasets = {};

function register(name, arg1, arg2) {
  if (typeof arg1 == "string") {
    datasets[name] = { load: (context) => fetchData(arg1, arg2, context) };
  } else {
    datasets[name] = { load: arg1 };
  }
}

// https://vega.github.io/vega-datasets/
// License: BSD-3-Clause
register(
  "movies",
  "https://cdn.jsdelivr.net/npm/vega-datasets@3.2.1/data/movies.json",
  "e63c499759e3b07b49563e036f55290f87feb56def8703ec049ca305ab1523d3",
);

// https://huggingface.co/datasets/scikit-learn/adult-census-income
// License: Creative Commons Zero v1.0 Universal
register(
  "census-income",
  "https://huggingface.co/datasets/scikit-learn/adult-census-income/resolve/fbeef6ec0e6fd88a5028b94683144000a6b380d5/adult.csv",
  "250e154ed75714ae57a564926d66c6319cd6aac1bcd32774cc76841a88d74e53",
);

// https://huggingface.co/datasets/gvlassis/california_housing
// License: MIT
register(
  "california-housing",
  "https://huggingface.co/datasets/gvlassis/california_housing/resolve/17110e600427385a20e8358b00adc9b032d185ff/data/train-00000-of-00001.parquet",
  "af9def52e328b6d6b661fe2ce6b2b2090094dfff9a4b2a7650adbe37cb29c3bd",
);

// https://huggingface.co/datasets/spawn99/wine-reviews
// License: CC BY-NC-SA 4.0
register("wine-reviews", async (context) => {
  let [{ data }, precomputed] = await Promise.all([
    fetchData(
      "https://huggingface.co/datasets/spawn99/wine-reviews/resolve/e6b10f4db3091a6fed8c5b294c0cc885e7f6e99d/data/train-00000-of-00001.parquet",
      "206876718697536f44129cfb3024ff0667a55f611fd8757a2be56048ea3280d3",
      context,
    ),
    context.fetch("../cache/20687671.parquet"),
  ]);
  /*!
    generate_dataset_embedding({
      "url": "https://huggingface.co/datasets/spawn99/wine-reviews/resolve/e6b10f4db3091a6fed8c5b294c0cc885e7f6e99d/data/train-00000-of-00001.parquet",
      "sha256": "206876718697536f44129cfb3024ff0667a55f611fd8757a2be56048ea3280d3",
      "query": "SELECT row_number() OVER (ORDER BY md5(description)) AS id, description AS text FROM data_frame GROUP BY title, country, province, description, points, price, variety, designation ORDER BY md5(description)",
      "output": "20687671.parquet"
    })
   */
  await context.db.registerFileBuffer("precomputed.parquet", precomputed);
  await context.db.registerFileBuffer("dataset.parquet", data);
  await context.connection.query(`
    CREATE TABLE ${context.table} AS
    SELECT
      title, country, province, description, points, price, variety, designation,
      x AS projection_x, y AS projection_y, neighbors
    FROM (
      SELECT row_number() OVER (ORDER BY md5(description)) AS id, title, country, province, description, points, price, variety, designation
      FROM 'dataset.parquet'
      GROUP BY title, country, province, description, points, price, variety, designation
      ORDER BY md5(description)
    ) AS dataset
    LEFT JOIN 'precomputed.parquet' AS precomputed ON dataset.id = precomputed.id
  `);
  await context.db.dropFile("dataset.parquet");
  await context.db.dropFile("precomputed.parquet");
  return true;
});

// https://huggingface.co/datasets/openlifescienceai/medmcqa
// License: Apache License 2.0
register("medmcqa", async (context) => {
  let [{ data }, precomputed] = await Promise.all([
    fetchData(
      "https://huggingface.co/datasets/openlifescienceai/medmcqa/resolve/91c6572c454088bf71b679ad90aa8dffcd0d5868/data/train-00000-of-00001.parquet",
      "b119434ba551517a6ec0ba1f7e0b4c029165ed284a4704f262ce37c791c493c5",
      context,
    ),
    context.fetch("../cache/b119434b.parquet"),
  ]);
  /*!
    generate_dataset_embedding({
      "url": "https://huggingface.co/datasets/openlifescienceai/medmcqa/resolve/91c6572c454088bf71b679ad90aa8dffcd0d5868/data/train-00000-of-00001.parquet",
      "sha256": "b119434ba551517a6ec0ba1f7e0b4c029165ed284a4704f262ce37c791c493c5",
      "query": "SELECT row_number() OVER () AS id, question AS text FROM data_frame",
      "output": "b119434b.parquet"
    })
   */
  await context.db.registerFileBuffer("precomputed.parquet", precomputed);
  await context.db.registerFileBuffer("dataset.parquet", data);
  await context.connection.query(`
    CREATE TABLE ${context.table} AS
    SELECT
      dataset.* EXCLUDE (__id__), x AS projection_x, y AS projection_y, neighbors
    FROM (SELECT dataset.*, row_number() OVER () AS __id__ FROM 'dataset.parquet') AS dataset
    LEFT JOIN 'precomputed.parquet' AS precomputed ON dataset.__id__ = precomputed.id
  `);
  await context.db.dropFile("dataset.parquet");
  await context.db.dropFile("precomputed.parquet");
  return true;
});

// https://huggingface.co/datasets/m-a-p/SuperGPQA
// License: Open Data Commons License Attribution family
register("SuperGPQA", async (context) => {
  let [{ data }, precomputed] = await Promise.all([
    fetchData(
      "https://huggingface.co/datasets/m-a-p/SuperGPQA/resolve/4430d4458112c7d4497fdcf94d7cc223313d6acf/SuperGPQA-all.jsonl",
      "28b998e70205ee95e540317b5adc06a06552a3961fb50b153df126b833f7a910",
      context,
    ),
    context.fetch("../cache/28b998e7.parquet"),
  ]);
  /*!
    generate_dataset_embedding({
      "url": "https://huggingface.co/datasets/m-a-p/SuperGPQA/resolve/4430d4458112c7d4497fdcf94d7cc223313d6acf/SuperGPQA-all.jsonl",
      "sha256": "28b998e70205ee95e540317b5adc06a06552a3961fb50b153df126b833f7a910",
      "query": "SELECT row_number() OVER () AS id, question AS text FROM data_frame",
      "output": "28b998e7.parquet"
    })
   */
  await context.db.registerFileBuffer("precomputed.parquet", precomputed);
  await context.db.registerFileBuffer("dataset.jsonl", data);
  await context.connection.query(`
    CREATE TABLE ${context.table} AS
    SELECT
      dataset.* EXCLUDE (__id__), x AS projection_x, y AS projection_y, neighbors
    FROM (SELECT dataset.*, row_number() OVER () AS __id__ FROM 'dataset.jsonl') AS dataset
    LEFT JOIN 'precomputed.parquet' AS precomputed ON dataset.__id__ = precomputed.id
  `);
  await context.db.dropFile("dataset.parquet");
  await context.db.dropFile("precomputed.parquet");
  return true;
});

// https://www.vispubdata.org/
// License: CC BY-NC 4.0
register("vispubdata", async (context) => {
  let [{ data }, precomputed] = await Promise.all([
    fetchData(
      "https://raw.githubusercontent.com/pisenberg/vispubdata/46b3a2b6f3cffa3a60dd76153c4b58704b286fda/vispubdata-update/results/vispubdata-update.csv-aminer.csv",
      "41ff9f2461ee9f9e24da6acc7e5aca1c0e5f11623c9cc0ba3b095930d23b563b",
      context,
    ),
    context.fetch("../cache/41ff9f24.parquet"),
  ]);
  /*!
    generate_dataset_embedding({
      "url": "https://raw.githubusercontent.com/pisenberg/vispubdata/46b3a2b6f3cffa3a60dd76153c4b58704b286fda/vispubdata-update/results/vispubdata-update.csv-aminer.csv",
      "sha256": "41ff9f2461ee9f9e24da6acc7e5aca1c0e5f11623c9cc0ba3b095930d23b563b",
      "query": "SELECT row_number() OVER () AS id, Abstract AS text FROM data_frame",
      "output": "41ff9f24.parquet"
    })
   */
  await context.db.registerFileBuffer("precomputed.parquet", precomputed);
  await context.db.registerFileBuffer("dataset.csv", data);
  await context.connection.query(`
    CREATE TEMPORARY MACRO clean_split(str, sep) AS list_filter(list_transform(string_split(str, ';'), x -> trim(x)), x -> x <> '');
    CREATE TABLE ${context.table} AS
    SELECT
      dataset.* EXCLUDE (__id__), x AS projection_x, y AS projection_y, neighbors
    FROM (
      SELECT
        row_number() OVER () AS __id__,
        Year, Conference, Title, Abstract, Link,
        (CASE PaperType
          WHEN 'J' THEN 'Journal Paper'
          WHEN 'C' THEN 'Conference Paper'
          WHEN 'M' THEN 'Miscellaneous'
        END) AS PaperType,
        (CASE Award
          WHEN 'BP' THEN 'Best Paper'
          WHEN 'HM' THEN 'Honorable Mention'
          WHEN 'TT' THEN 'Test of Time'
          WHEN 'BA' THEN 'Best Application Paper'
          WHEN 'BCS' THEN 'Best Case Study'
          WHEN 'TT;BP' THEN 'Test of Time; Best Paper'
        END) AS Award,
        clean_split("AuthorNames-Deduped", ';') AS AuthorNames_Deduped,
        clean_split("AuthorAffiliation", ';') AS AuthorAffiliation,
        clean_split("AuthorKeywords", ',') AS AuthorKeywords,
        AminerCitationCount,
        CitationCount_CrossRef,
        PubsCited_CrossRef
      FROM 'dataset.csv'
    ) AS dataset
     LEFT JOIN 'precomputed.parquet' AS precomputed ON dataset.__id__ = precomputed.id
  `);
  await context.db.dropFile("dataset.csv");
  await context.db.dropFile("precomputed.parquet");
  return true;
});

// https://huggingface.co/datasets/derek-thomas/ScienceQA
// License: Creative Commons Attribution Share Alike 4.0
register(
  "ScienceQA",
  "https://huggingface.co/datasets/derek-thomas/ScienceQA/resolve/f18b0a70359ebfb41f658fd564208d0355b013f4/data/train-00000-of-00001-1028f23e353fbe3e.parquet",
  "62c90a28e3fb1bc0ad7bbcab1ac62b483ae6758291a655944d8f494bf6445745",
);

async function fetchData(url, sha1sum, context) {
  let data = await context.fetch(url);
  let digest = await window.crypto.subtle.digest("SHA-256", data);
  let digestStr = toHex(new Uint8Array(digest));
  if (digestStr != sha1sum) {
    throw new Error(`Error: data integrity check failed, maybe the data has been modified (sha256: ${digestStr})`);
  }
  return { data: data, filename: url };
}

function toHex(bytes) {
  let hex = "";
  for (let i = 0; i < bytes.length; i++) {
    hex += bytes[i].toString(16).padStart(2, "0");
  }
  return hex;
}

async function loadDataFromUrl(url, context) {
  if (url.startsWith("example://")) {
    let name = url.substring("example://".length);
    return datasets[name].load(context);
  }
  return undefined;
}

export { loadDataFromUrl };
