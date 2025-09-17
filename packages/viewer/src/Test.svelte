<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import type { Coordinator } from "@uwdata/mosaic-core";

  import Viewer from "./Viewer.svelte";

  import type { DataSource } from "./data_source.js";
  import type { EmbeddingAtlasProps } from "./lib/api.js";
  import { initializeDatabase } from "./lib/database_utils.js";

  export class TestDataSource implements DataSource {
    private count: number;

    constructor(count: number) {
      this.count = count;
    }

    async initializeCoordinator(coordinator: Coordinator, table: string): Promise<Partial<EmbeddingAtlasProps>> {
      await initializeDatabase(coordinator, "wasm");

      await coordinator.exec(`
        SELECT setseed(0.42);

        CREATE OR REPLACE MACRO random_normal() AS (sqrt(-2 * LN(random())) * cos(2 * PI() * random()));
        CREATE OR REPLACE MACRO random_choice(list, f) AS list_element(list, 1 + floor(pow(random(), f) * len(list))::INT);
        CREATE OR REPLACE MACRO random_invalid(x) AS CASE WHEN random() < 0.02 THEN random_choice(['NaN','Inf','-Inf', Null], 2)::DOUBLE ELSE x END;
        CREATE OR REPLACE MACRO random_string(prefix) AS prefix || LPAD(FLOOR(POW(RANDOM(), 1.3) * 1000)::INT::VARCHAR, 3, '0');
        CREATE OR REPLACE MACRO shuffle_hash(x, v) AS x + hash(v) % 5;

        CREATE OR REPLACE TABLE ${table} AS (
          SELECT
            id,
            'sample text' AS text,
            random_choice(['Apple', 'Orange', 'Pink-flowered Native Raspberry', null], 2) AS var_category,
            concat('Category', (150 * abs(sin(20 * pow(random(), 5))))::INT::TEXT) AS var_many_category,
            list_distinct(list_transform(range(1, 4), (x) -> random_choice(['Apple', 'Orange', 'Banana', 'Raspberry', 'Pear'], 2))) AS var_multi_category,
            list_distinct(list_transform(range(1, 4), (x) -> random_string('Cat'))) AS var_multi_many_category,
            random() AS var_uniform,
            random_invalid(random_normal()) AS var_normal,
            random_invalid(exp(random_normal() + 1)) AS var_log_normal,
            random_invalid(random_normal() + 1000) AS var_normal_biased,
            random_invalid(random_normal() - 1000) AS var_normal_biased_negative,
            random_normal() AS x,
            random_normal() AS y
          FROM range(0, ${this.count}) t(id)
        );

        UPDATE ${table} SET var_normal = var_normal + 5 WHERE var_log_normal < exp(random());
        UPDATE ${table} SET var_normal_biased = var_normal_biased + random() * var_normal;
        UPDATE ${table} SET var_normal_biased_negative = var_normal_biased_negative + random() * var_normal;
        UPDATE ${table} SET
          x = sin(10 * shuffle_hash(var_uniform, var_category)) * x + cos(5 * shuffle_hash(var_uniform, var_category)) * y,
          y = sin(4 * shuffle_hash(var_uniform, var_category)) * x + cos(18 * shuffle_hash(var_uniform, var_category)) * y;
        UPDATE ${table} SET x = x + 5 * fmod(floor(var_uniform * 24 + random()), 5);
        UPDATE ${table} SET y = y + 5 * floor(floor(var_uniform * 24 + random()) / 5);
      `);

      return {
        data: {
          table: table,
          id: "id",
          projection: { x: "x", y: "y" },
          text: "text",
        },
      };
    }
  }

  let dataSource = new TestDataSource(100000);
</script>

<Viewer dataSource={dataSource} />
