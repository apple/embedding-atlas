// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { getContext, setContext } from "svelte";
import { HorizontalScrollbarController } from "../controllers/HorizontalScrollbarController.svelte";
import { TableController } from "../controllers/TableController.svelte";
import { TablePortalController } from "../controllers/TablePortalController.svelte";
import { VerticalScrollbarController } from "../controllers/VerticalScrollbarController.svelte";
import { Schema } from "../model/Schema.svelte";
import { TableModel } from "../model/TableModel.svelte";
import { OverscrollModifier } from "../modifiers/overscroll.svelte";

const SCHEMA_KEY = Symbol("schema");
const MODEL_KEY = Symbol("model");
const CONTROLLER_KEY = Symbol("controller");
const VERTICAL_SCROLLBAR_CONTROLLER_KEY = Symbol("vertical-scrollbar-controller");
const HORIZONTAL_SCROLLBAR_CONTROLLER_KEY = Symbol("horizontal-scrollbar-controller");
const TABLE_PORTAL_CONTROLLER_KEY = Symbol("table-portal-controller");
const OVERSCROLL_MODIFIER_KEY = Symbol("overscroll-modifier");

export class Context {
  public static initialize() {
    const schema = new Schema();
    const model = new TableModel(schema);
    const controller = new TableController(model, schema);
    const verticalScrollbarController = new VerticalScrollbarController({
      tableModel: model,
      tableController: controller,
    });
    const horizontalScrollbarContainer = new HorizontalScrollbarController({
      tableModel: model,
      tableController: controller,
    });
    const tablePortalController = new TablePortalController(controller);
    const overscrollModifier = new OverscrollModifier(controller);

    setContext(SCHEMA_KEY, schema);
    setContext(MODEL_KEY, model);
    setContext(CONTROLLER_KEY, controller);
    setContext(VERTICAL_SCROLLBAR_CONTROLLER_KEY, verticalScrollbarController);
    setContext(HORIZONTAL_SCROLLBAR_CONTROLLER_KEY, horizontalScrollbarContainer);
    setContext(TABLE_PORTAL_CONTROLLER_KEY, tablePortalController);
    setContext(OVERSCROLL_MODIFIER_KEY, overscrollModifier);
  }

  public static get schema(): Schema {
    return getContext(SCHEMA_KEY);
  }

  public static get model(): TableModel {
    return getContext(MODEL_KEY);
  }

  public static get controller(): TableController {
    return getContext(CONTROLLER_KEY);
  }

  public static get verticalScrollbarController(): VerticalScrollbarController {
    return getContext(VERTICAL_SCROLLBAR_CONTROLLER_KEY);
  }

  public static get horizontalScrollbarController(): HorizontalScrollbarController {
    return getContext(HORIZONTAL_SCROLLBAR_CONTROLLER_KEY);
  }

  public static get tablePortalController(): TablePortalController {
    return getContext(TABLE_PORTAL_CONTROLLER_KEY);
  }

  public static get overscrollModifier(): OverscrollModifier {
    return getContext(OVERSCROLL_MODIFIER_KEY);
  }
}
